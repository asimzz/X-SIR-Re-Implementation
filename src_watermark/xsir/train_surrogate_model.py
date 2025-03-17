import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import argparse
from torch.optim import lr_scheduler
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(dim, dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc(x)
        out = self.relu(out)
        out = out + x
        return out


class TransformModel(nn.Module):
    def __init__(self, num_layers=4, input_dim=1024, hidden_dim=500, output_dim=300):
        super(TransformModel, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(ResidualBlock(hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        # Convert x to the same data type as the first layer
        layer_dtype = self.layers[0].weight.dtype
        x = x.to(layer_dtype)
        for layer in self.layers:
            x = layer(x)
        return x


def cosine_similarity(x, y):
    dot_product = torch.sum(x * y, dim=-1)
    norm_x = torch.norm(x, p=2, dim=-1)
    norm_y = torch.norm(y, p=2, dim=-1)
    return dot_product / (norm_x * norm_y)


def row_col_mean_penalty(output):
    row_mean_penalty = torch.mean(output, dim=1).pow(2).sum()
    col_mean_penalty = torch.mean(output, dim=0).pow(2).sum()
    return row_mean_penalty + col_mean_penalty


def abs_value_penalty(output):
    penalties = torch.relu(0.05 - torch.abs(output))
    mask = (penalties > 0).float()  # Create a mask where penalties are non-zero
    non_zero_count = torch.max(mask.sum(), torch.tensor(1.0))  # Avoid division by zero
    return (penalties * mask).sum() / non_zero_count


def vector_transform(vec):
    mean = vec.mean(dim=0, keepdim=True)
    centered_x = vec - mean
    transformed_x = torch.tanh(30 * centered_x)
    return transformed_x


def loss_fn(
    output_a, output_b, input_a, input_b, lambda1=0.1, lambda2=1, median_value=0.4
):
    original_similarity = cosine_similarity(input_a, input_b)
    original_similarity = torch.tanh(20 * (original_similarity - median_value))
    transformed_similarity = cosine_similarity(output_a, output_b)
    original_loss = torch.abs(original_similarity - transformed_similarity).mean()

    mean_penalty = row_col_mean_penalty(output_a) + row_col_mean_penalty(output_b)
    range_penalty_a = abs_value_penalty(output_a)
    range_penalty_b = abs_value_penalty(output_b)

    total_loss = (
        original_loss
        + lambda1 * mean_penalty
        + lambda2 * (range_penalty_a + range_penalty_b)
    )
    return total_loss


def cosine_similarity_matrix(batch):
    norm = torch.norm(batch, dim=1).view(-1, 1)
    normed_batch = batch / norm
    similarity = torch.mm(normed_batch, normed_batch.t())
    return similarity


def get_median_value_of_similarity(all_token_embedding):
    similarity = cosine_similarity_matrix(all_token_embedding)
    median_value = torch.median(similarity)
    print("Median similarity value:", median_value)
    return median_value


class VectorDataset(Dataset):
    def __init__(self, vectors):
        self.vectors = vectors

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, idx):
        return self.vectors[idx]


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train surrogate watermark models with different seeds"
    )
    parser.add_argument(
        "--input_path", type=str, default="embeddings/gpt2_sentence_embedding_bert.txt"
    )
    parser.add_argument("--output_model_prefix", type=str, default="transform_model")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.006)
    parser.add_argument("--input_dim", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[123, 456, 789],
        help="List of random seeds for surrogate training runs",
    )
    args = parser.parse_args()

    # Load the embedding data and compute median similarity
    embedding_data = np.loadtxt(args.input_path)
    data = torch.tensor(embedding_data, device=device, dtype=torch.float32)
    median_value = get_median_value_of_similarity(data)
    dataset = VectorDataset(data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Loop over each specified random seed
    for seed in args.seeds:
        print(f"\nStarting training for surrogate model with seed: {seed}")
        set_random_seed(seed)

        # Initialize the model, optimizer, and learning rate scheduler for the run
        model = TransformModel(input_dim=args.input_dim).to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.2)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)

        for epoch in range(args.epochs):
            batch_iterator = iter(dataloader)
            num_steps = len(dataloader) // 2  # Two batches per step
            for step in range(num_steps):
                try:
                    input_a = next(batch_iterator).to(device)
                    input_b = next(batch_iterator).to(device)
                except StopIteration:
                    break

                # Ensure both batches have the same size
                if input_a.shape[0] != input_b.shape[0]:
                    continue

                output_a = model(input_a)
                output_b = model(input_b)
                loss = loss_fn(
                    output_a, output_b, input_a, input_b, median_value=median_value
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step % 100 == 0:
                    print(
                        f"Seed {seed}, Epoch [{epoch+1}/{args.epochs}], Step [{step+1}/{num_steps}], Loss: {loss.item():.4f}"
                    )
            scheduler.step()

        # Save the trained surrogate model with a unique filename per seed
        output_model_name = f"{args.output_model_prefix}_seed_{seed}.pth"
        os.makedirs(os.path.dirname(output_model_name) or ".", exist_ok=True)
        torch.save(model.state_dict(), output_model_name)
        print(f"Model saved to {output_model_name}")
