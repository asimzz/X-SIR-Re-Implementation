from urielplus import urielplus
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
# Removed plotting imports - only CSV output needed

def create_genetic_distance_clusters():
    """Create language clusters using genetic distance only"""

    u = urielplus.URIELPlus()

    print("=== Genetic Distance Language Clustering ===")

    # Get list of available languages with genetic distance data
    all_languages = set(u.get_languages_with_distance_data(distance_type="genetic"))

    # Select 100 well-known languages, filtering for those with genetic data
    candidate_languages = [
        'eng', 'spa', 'fra', 'deu', 'ita', 'por', 'rus', 'jpn', 'zho', 'ara', 'hin', 'kor',
        'nld', 'pol', 'tur', 'vie', 'tha', 'swe', 'nor', 'dan', 'fin', 'hun', 'ces', 'slk',
        'ron', 'bul', 'hrv', 'srp', 'slv', 'est', 'lav', 'lit', 'ell', 'heb', 'fas', 'urd',
        'ben', 'tam', 'tel', 'kan', 'mal', 'guj', 'pan', 'mar', 'ori', 'asm', 'nep', 'sin',
        'mya', 'khm', 'lao', 'mon', 'bod', 'uig', 'kaz', 'kir', 'uzb', 'tgk', 'aze', 'kat',
        'hye', 'bel', 'ukr', 'ltz', 'mlt', 'isl', 'fao', 'gle', 'gla', 'cym', 'bre', 'eus',
        'cat', 'glg', 'ast', 'mwl', 'vec', 'lij', 'pms', 'lmo', 'cos', 'srd', 'scn', 'nap',
        'lad', 'arg', 'ext', 'mdf', 'myv', 'kpv', 'udm', 'krl', 'vep', 'izh', 'liv', 'sme',
        'smn', 'sms', 'smj', 'sma', 'afr', 'swa', 'hau', 'yor', 'ibo', 'amh', 'som', 'orm'
    ]

    # Filter to only languages that have genetic distance data
    languages = [lang for lang in candidate_languages if lang in all_languages]
    print(f"Found {len(languages)} valid languages with genetic data")
    print(f"Clustering {len(languages)} languages: {languages}")

    # Focus only on genetic distance
    distance_types = ['genetic']

    all_results = {}

    for distance_type in distance_types:
        print(f"\n=== {distance_type.upper()} Distance Clustering ===")

        # Build distance matrix
        print(f"Building {distance_type} distance matrix...")
        distance_matrix = build_simple_distance_matrix(u, languages, distance_type)

        if distance_matrix is not None:
            print(f"Distance matrix created successfully")
            print(f"Distance range: [{np.min(distance_matrix):.3f}, {np.max(distance_matrix):.3f}]")

            # Perform clustering
            results = cluster_with_distances(distance_matrix, languages, distance_type)

            if results:
                all_results[distance_type] = results

                # Show results
                best_result = results['best_clustering']
                print(f"✅ Best clustering: {best_result['n_clusters']} clusters, silhouette={best_result['silhouette']:.3f}")

                # Show cluster composition
                show_cluster_composition(languages, best_result['labels'], distance_type)

        else:
            print(f"❌ Failed to build distance matrix")

    # Save results to CSV files
    if all_results:
        save_results(all_results, languages)

        print(f"\n✅ Clustering completed successfully!")
        print(f"📊 Check generated CSV files for clustering results")

    return all_results

def build_simple_distance_matrix(u, languages, distance_type):
    """Build distance matrix with progress tracking"""

    n_langs = len(languages)
    distance_matrix = np.zeros((n_langs, n_langs))

    # Get distance function
    distance_functions = {
        'genetic': u.new_genetic_distance
    }

    distance_func = distance_functions[distance_type]

    total_pairs = n_langs * (n_langs - 1) // 2
    completed = 0

    print(f"  Calculating {total_pairs} distance pairs...")

    for i in range(n_langs):
        for j in range(i + 1, n_langs):
            try:
                lang1, lang2 = languages[i], languages[j]
                distance = distance_func([lang1, lang2])

                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

                completed += 1
                if completed % 10 == 0 or completed == total_pairs:
                    print(f"    Progress: {completed}/{total_pairs} ({(completed/total_pairs)*100:.0f}%)")

            except Exception as e:
                print(f"    Error calculating distance between {languages[i]} and {languages[j]}: {e}")
                distance_matrix[i, j] = 1.0
                distance_matrix[j, i] = 1.0

    return distance_matrix

def cluster_with_distances(distance_matrix, languages, distance_type):
    """Perform hierarchical clustering with different cluster numbers"""

    n_languages = len(languages)
    results = {}

    print(f"  Testing different cluster numbers...")

    # Try different numbers of clusters - scale with number of languages
    max_clusters = min(n_languages // 2, n_languages - 1, 30)  # Cap at 30 for performance
    cluster_range = range(2, max_clusters + 1)
    best_silhouette = -1
    best_result = None

    for n_clusters in cluster_range:
        try:
            # Hierarchical clustering
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='precomputed',
                linkage='average'
            )

            labels = clustering.fit_predict(distance_matrix)

            # Calculate silhouette score
            silhouette_avg = silhouette_score(distance_matrix, labels, metric='precomputed')

            results[f'hierarchical_{n_clusters}'] = {
                'labels': labels,
                'n_clusters': n_clusters,
                'silhouette': silhouette_avg,
                'algorithm': f'Hierarchical (k={n_clusters})'
            }

            print(f"    k={n_clusters}: silhouette={silhouette_avg:.3f}")

            if silhouette_avg > best_silhouette:
                best_silhouette = silhouette_avg
                best_result = results[f'hierarchical_{n_clusters}']

        except Exception as e:
            print(f"    k={n_clusters}: Error - {e}")

    if best_result:
        return {
            'all_results': results,
            'best_clustering': best_result,
            'distance_matrix': distance_matrix,
            'distance_type': distance_type
        }
    else:
        return None

def show_cluster_composition(languages, labels, distance_type):
    """Display cluster composition"""

    print(f"\n{distance_type.upper()} Cluster Composition:")
    print("-" * 40)

    # Group languages by cluster
    clusters = {}
    for lang, cluster_id in zip(languages, labels):
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(lang)

    for cluster_id, cluster_langs in sorted(clusters.items()):
        print(f"Cluster {cluster_id}: {', '.join(cluster_langs)}")

# Visualization function removed - only CSV output needed

def save_results(results, languages):
    """Save clustering results"""

    for distance_type, result in results.items():
        best_clustering = result['best_clustering']

        # Save cluster assignments
        df = pd.DataFrame({
            'language': languages,
            'cluster_id': best_clustering['labels'],
            'distance_type': distance_type,
            'clustering_method': best_clustering['algorithm']
        })

        filename = f"clusters_{distance_type}.csv"
        df.to_csv(filename, index=False)
        print(f"Saved {distance_type} clusters to '{filename}'")

        # Save distance matrix
        distance_df = pd.DataFrame(
            result['distance_matrix'],
            index=languages,
            columns=languages
        )

        distance_filename = f"distances_{distance_type}.csv"
        distance_df.to_csv(distance_filename)
        print(f"Saved {distance_type} distances to '{distance_filename}'")

# Main execution
if __name__ == "__main__":
    results = create_genetic_distance_clusters()