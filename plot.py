import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data from your file (replace with your actual file path)
file_path = 'Book1.csv'  # Update this path

# Read the CSV file
df = pd.read_csv(file_path, delimiter=';')
df.columns = df.columns.str.strip()

# Unique models and pivot languages
models = df['Model'].unique()
languages = df['Pivot Language'].unique()

# Set up the radar chart
num_vars = len(languages)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # Complete the loop

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Define a color palette
colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

# Normalize the AUC values to enhance separation
scale_factor = 1  # Adjust this factor to scale the AUC values

# Plot each model on the radar chart
for i, model in enumerate(models):
    # Filter data for the current model
    model_data = df[df['Model'] == model]
    auc_values = []

    # Ensure the order of pivot languages is consistent
    for lang in languages:
        auc = model_data.loc[model_data['Pivot Language'] == lang, 'AUC']
        auc_values.append(auc.values[0] * scale_factor if not auc.empty else 0)

    # Close the loop for the radar chart
    auc_values += auc_values[:1]

    # Plot the radar chart for the current model
    ax.plot(angles, auc_values, color=colors[i], linewidth=2, marker='o', label=model)
    ax.fill(angles, auc_values, color=colors[i], alpha=0.25)

# Add labels for each axis
ax.set_xticks(angles[:-1])
ax.set_xticklabels(languages, fontsize=12)

# Increase the radial spacing for better value separation
ax.set_yticks(np.linspace(0, scale_factor, 20))
ax.set_yticklabels([f'{x/scale_factor:.2f}' for x in np.linspace(0, scale_factor, 20)])

# Add title and legend
plt.title('Radar Chart: Enhanced AUC Comparison Across Pivot Languages', size=14, pad=20)
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1), title="Models")

# Display the plot
plt.show()
