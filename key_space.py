import argparse
import json
from src_watermark.xsir.key_space import XSIRKeySpace
from src_watermark.xsir.key_space import KGWKeySpace


def print_key_space_size(key_space_size):
    coefficient, exponent = key_space_size
    print(f"Key space size: {coefficient:.3f} * 2^{exponent}")


def main(args):
    if args.split_type == "random":
        key_space = KGWKeySpace(args.mapping_file, args.gamma)
        key_space_size = key_space.compute_key_space_size()
        print_key_space_size(key_space_size)
    if args.split_type == "semantic":
        key_space = XSIRKeySpace(args.mapping_file, args.cluster_file)
        key_space_size = key_space.compute_key_space_size()
        print(f"Cluster size: {key_space.cluster_size}")
        print_key_space_size(key_space_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate with watermarking")

    parser.add_argument(
        "--cluster_file",
        type=str,
        required=True,
        help="Cluster file containing the cluster groups per model",
    )
    parser.add_argument(
        "--mapping_file",
        type=str,
        required=True,
        help="Mapping file containing the cluster of each token in the vocabulary",
    )

    parser.add_argument(
        "--split_type", type=str, default="random", help="Either random or semantic"
    )
    parser.add_argument("--gamma", type=float, default=0.5)

    args = parser.parse_args()

    main(args)
