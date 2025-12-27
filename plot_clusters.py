#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import argparse

def load_cluster_data(clusters_file='clusters_genetic.csv', distances_file='distances_genetic.csv'):
    """Load cluster assignments and distance matrix"""

    # Load cluster assignments
    clusters_df = pd.read_csv(clusters_file)

    # Load distance matrix
    distances_df = pd.read_csv(distances_file, index_col=0)

    print(f"Loaded {len(clusters_df)} languages with {clusters_df['cluster_id'].nunique()} clusters")
    print(f"Distance matrix shape: {distances_df.shape}")

    return clusters_df, distances_df

def create_scatter_plot_tsne(clusters_df, distances_df, output_file='clusters_scatter_tsne.png'):
    """Create scatter plot using t-SNE dimensionality reduction"""

    print("Creating t-SNE scatter plot...")

    # Prepare data
    languages = clusters_df['language'].tolist()
    cluster_labels = clusters_df['cluster_id'].tolist()

    # Ensure distance matrix matches language order
    distance_matrix = distances_df.loc[languages, languages].values

    # Apply t-SNE
    tsne = TSNE(n_components=2, metric='precomputed', random_state=42,
                perplexity=min(30, len(languages)//3), init='random')
    tsne_coords = tsne.fit_transform(distance_matrix)

    # Create scatter plot with better styling
    plt.figure(figsize=(16, 12))
    plt.style.use('default')

    # Get unique clusters and use very distinct colors
    unique_clusters = sorted(set(cluster_labels))

    # Use highly contrasting colors - combine multiple color schemes
    import matplotlib.colors as mcolors

    # Create a list of very distinct colors
    distinct_colors = [
        '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',  # Set1 colors
        '#ffff33', '#a65628', '#f781bf', '#999999', '#8dd3c7',  # More distinct
        '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462',  # Set3 colors
        '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd', '#ccebc5',  # Pastel2
        '#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e'   # Dark2 colors
    ]

    # Ensure we have enough colors
    while len(distinct_colors) < len(unique_clusters):
        distinct_colors.extend(distinct_colors)

    colors = distinct_colors[:len(unique_clusters)]

    # Create cluster size mapping for different marker sizes
    cluster_sizes = {cluster_id: sum(1 for c in cluster_labels if c == cluster_id)
                     for cluster_id in unique_clusters}

    # Plot each cluster with different styling based on size
    for i, cluster_id in enumerate(unique_clusters):
        mask = np.array(cluster_labels) == cluster_id
        cluster_langs = [lang for j, lang in enumerate(languages) if mask[j]]
        cluster_size = len(cluster_langs)

        # Adjust marker size based on cluster size
        marker_size = max(40, min(120, 40 + cluster_size * 3))

        # Adjust alpha based on cluster size (larger clusters more transparent)
        alpha = max(0.5, 1.0 - cluster_size / 25)

        plt.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1],
                   c=colors[i], label=f'Cluster {cluster_id} ({cluster_size}L)',
                   alpha=alpha, s=marker_size, edgecolors='black', linewidth=0.5)

        # NO language labels on the plot - keep it clean

    # Enhanced styling
    plt.title('Language Genetic Clustering (t-SNE Projection)\n108 Languages in 22 Clusters',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)

    # Better legend with multiple columns
    legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2,
                       frameon=True, fancybox=True, shadow=True,
                       title='Clusters (Languages)', title_fontsize=12)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)

    # Add grid for better readability
    plt.grid(True, alpha=0.3, linestyle='--')

    # Add text annotation for context
    plt.text(0.02, 0.98, 'Based on URIEL+ Genetic Distance',
             transform=plt.gca().transAxes, fontsize=10, alpha=0.7,
             verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved enhanced t-SNE scatter plot to {output_file}")
    plt.show()

def create_scatter_plot_pca(clusters_df, distances_df, output_file='clusters_scatter_pca.png'):
    """Create scatter plot using PCA dimensionality reduction"""

    print("Creating PCA scatter plot...")

    # Prepare data
    languages = clusters_df['language'].tolist()
    cluster_labels = clusters_df['cluster_id'].tolist()

    # Ensure distance matrix matches language order
    distance_matrix = distances_df.loc[languages, languages].values

    # Convert distances to similarities for PCA
    max_dist = np.max(distance_matrix)
    similarity_matrix = max_dist - distance_matrix

    # Apply PCA
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(similarity_matrix)

    # Create scatter plot
    plt.figure(figsize=(12, 8))

    # Get unique clusters and colors
    unique_clusters = sorted(set(cluster_labels))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))

    # Plot each cluster
    for i, cluster_id in enumerate(unique_clusters):
        mask = np.array(cluster_labels) == cluster_id
        cluster_langs = [lang for j, lang in enumerate(languages) if mask[j]]

        plt.scatter(pca_coords[mask, 0], pca_coords[mask, 1],
                   c=[colors[i]], label=f'Cluster {cluster_id} ({len(cluster_langs)})',
                   alpha=0.7, s=60)

        # Add language labels (only for smaller clusters to avoid clutter)
        if len(cluster_langs) <= 5:  # Only label small clusters
            for j, (x, y) in enumerate(pca_coords[mask]):
                plt.annotate(cluster_langs[j], (x, y), xytext=(3, 3),
                            textcoords='offset points', fontsize=7, alpha=0.8)

    plt.title(f'Language Clusters (PCA Projection)\nPC1: {pca.explained_variance_ratio_[0]:.1%}, PC2: {pca.explained_variance_ratio_[1]:.1%} variance')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved PCA scatter plot to {output_file}")
    plt.show()

def create_heatmap_by_clusters(clusters_df, distances_df, output_file='clusters_heatmap.png'):
    """Create heatmap with languages ordered by cluster"""

    print("Creating cluster-ordered heatmap...")

    # Sort languages by cluster
    clusters_df_sorted = clusters_df.sort_values(['cluster_id', 'language'])
    languages_ordered = clusters_df_sorted['language'].tolist()

    # Reorder distance matrix
    distance_matrix_ordered = distances_df.loc[languages_ordered, languages_ordered]

    # Create cluster boundaries for visualization
    cluster_boundaries = []
    current_cluster = clusters_df_sorted.iloc[0]['cluster_id']

    for i, row in clusters_df_sorted.iterrows():
        if row['cluster_id'] != current_cluster:
            cluster_boundaries.append(clusters_df_sorted.index.get_loc(i))
            current_cluster = row['cluster_id']

    # Create heatmap
    plt.figure(figsize=(15, 12))

    sns.heatmap(distance_matrix_ordered,
                xticklabels=languages_ordered,
                yticklabels=languages_ordered,
                cmap='viridis',
                square=True,
                cbar_kws={'label': 'Genetic Distance'})

    # Add cluster boundaries
    for boundary in cluster_boundaries:
        plt.axhline(y=boundary, color='red', linewidth=2, alpha=0.7)
        plt.axvline(x=boundary, color='red', linewidth=2, alpha=0.7)

    plt.title('Genetic Distance Matrix (Ordered by Clusters)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved cluster heatmap to {output_file}")
    plt.show()

def print_cluster_summary(clusters_df):
    """Print summary of clusters"""

    print("\n=== Cluster Summary ===")

    for cluster_id in sorted(clusters_df['cluster_id'].unique()):
        cluster_langs = clusters_df[clusters_df['cluster_id'] == cluster_id]['language'].tolist()
        print(f"Cluster {cluster_id} ({len(cluster_langs)} languages): {', '.join(cluster_langs)}")

    print(f"\nTotal: {len(clusters_df)} languages in {clusters_df['cluster_id'].nunique()} clusters")

def create_cluster_overview(clusters_df, output_file='cluster_overview.png'):
    """Create overview chart showing cluster sizes and composition"""

    print("Creating cluster overview...")

    # Get cluster sizes
    cluster_sizes = clusters_df.groupby('cluster_id').size().sort_values(ascending=False)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Bar plot of cluster sizes
    cluster_sizes.plot(kind='bar', ax=ax1, color='skyblue', alpha=0.8)
    ax1.set_title('Languages per Cluster')
    ax1.set_xlabel('Cluster ID')
    ax1.set_ylabel('Number of Languages')
    ax1.tick_params(axis='x', rotation=45)

    # Pie chart of cluster distribution
    ax2.pie(cluster_sizes.values, labels=[f'C{i}' for i in cluster_sizes.index],
            autopct='%1.0f%%', startangle=90)
    ax2.set_title('Cluster Size Distribution')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved cluster overview to {output_file}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize language clusters')
    parser.add_argument('--clusters', default='clusters_genetic.csv', help='Clusters CSV file')
    parser.add_argument('--distances', default='distances_genetic.csv', help='Distances CSV file')
    parser.add_argument('--plot-type', choices=['tsne', 'pca', 'heatmap', 'overview', 'all'], default='tsne',
                       help='Type of visualization to create (for 100 langs, recommend tsne or overview)')

    args = parser.parse_args()

    # Load data
    clusters_df, distances_df = load_cluster_data(args.clusters, args.distances)

    # Print cluster summary
    print_cluster_summary(clusters_df)

    # Create visualizations
    if args.plot_type in ['overview', 'all']:
        create_cluster_overview(clusters_df)

    if args.plot_type in ['tsne', 'all']:
        create_scatter_plot_tsne(clusters_df, distances_df)

    if args.plot_type in ['pca', 'all']:
        create_scatter_plot_pca(clusters_df, distances_df)

    if args.plot_type in ['heatmap', 'all'] and len(clusters_df) <= 50:
        print("Note: Skipping heatmap for >50 languages (too cluttered)")
        create_heatmap_by_clusters(clusters_df, distances_df)
    elif args.plot_type == 'heatmap' and len(clusters_df) > 50:
        print("Heatmap not recommended for >50 languages. Use 'overview' or 'tsne' instead.")

    print("\n✅ Visualization complete!")

if __name__ == "__main__":
    main()