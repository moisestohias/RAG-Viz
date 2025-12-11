"""
UMAP Visualization Script for RAG Project
This script loads document embeddings and creates 2D and 3D visualizations using different UMAP parameters.
"""
import json
import numpy as np
import umap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import os
from helper_utils import load_data_from_db

def load_data(data_path="data/data.json", embeddings_path="data/doc_emb.json"):
    """Load data and embeddings from JSON files"""
    print("Loading data...")

    # Load content data
    with open(data_path, 'r') as f:
        data = json.load(f)

    # Load embeddings
    with open(embeddings_path, 'r') as f:
        embeddings = json.load(f)

    # Match embeddings with data based on keys (file paths)
    valid_keys = [k for k in data.keys() if k in embeddings]
    print(f"Found {len(valid_keys)} matching entries")

    # Extract embeddings and content
    embedding_vectors = [embeddings[k] for k in valid_keys]
    content_snippets = [data[k]['content'] for k in valid_keys]

    return np.array(embedding_vectors, dtype=np.float32), content_snippets, valid_keys

def standardize_embeddings(embeddings):
    """Standardize embeddings to have zero mean and unit variance"""
    print("Standardizing embeddings...")
    scaler = StandardScaler()
    standardized_embeddings = scaler.fit_transform(embeddings)
    return standardized_embeddings, scaler

def create_umap_plots(embeddings, labels, plot_settings, output_dir="plots"):
    """Create UMAP plots with different settings"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define UMAP configurations
    umap_configs = [
        {
            "name": "default_params",
            "params": {
                "n_neighbors": 15,
                "min_dist": 0.1,
                "metric": "euclidean",
                "random_state": 42
            }
        },
        {
            "name": "local_structure",
            "params": {
                "n_neighbors": 5,
                "min_dist": 0.01,
                "metric": "euclidean",
                "random_state": 42
            }
        },
        {
            "name": "global_structure",
            "params": {
                "n_neighbors": 50,
                "min_dist": 0.5,
                "metric": "euclidean",
                "random_state": 42
            }
        },
        {
            "name": "cosine_metric",
            "params": {
                "n_neighbors": 15,
                "min_dist": 0.1,
                "metric": "cosine",
                "random_state": 42
            }
        },
        {
            "name": "clustering_friendly",
            "params": {
                "n_neighbors": 30,
                "min_dist": 0.0,
                "metric": "euclidean",
                "random_state": 42
            }
        }
    ]
    
    # For each configuration, create both 2D and 3D plots
    for config in umap_configs:
        print(f"\nGenerating plots for configuration: {config['name']}")
        
        # 2D plot
        print(f"  - Creating 2D plot...")
        reducer_2d = umap.UMAP(
            n_components=2,
            **config["params"]
        )
        embedding_2d = reducer_2d.fit_transform(embeddings)
        
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                            c=range(len(embedding_2d)), cmap='tab20', s=5)
        plt.colorbar(scatter)
        plt.title(f'UMAP 2D Projection\nConfiguration: {config["name"]}\n'
                  f'n_neighbors={config["params"]["n_neighbors"]}, '
                  f'min_dist={config["params"]["min_dist"]}, '
                  f'metric={config["params"]["metric"]}')
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        plt.tight_layout()
        
        filename_2d = os.path.join(output_dir, f'umap_2d_{config["name"]}.png')
        plt.savefig(filename_2d, dpi=150, bbox_inches='tight')
        print(f"    Saved: {filename_2d}")
        plt.close()
        
        # 3D plot
        print(f"  - Creating 3D plot...")
        reducer_3d = umap.UMAP(
            n_components=3,
            **config["params"]
        )
        embedding_3d = reducer_3d.fit_transform(embeddings)
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(embedding_3d[:, 0], 
                           embedding_3d[:, 1], 
                           embedding_3d[:, 2], 
                           c=range(len(embedding_3d)), 
                           cmap='tab20', s=5)
        ax.set_xlabel('UMAP Component 1')
        ax.set_ylabel('UMAP Component 2')
        ax.set_zlabel('UMAP Component 3')
        ax.set_title(f'UMAP 3D Projection\nConfiguration: {config["name"]}\n'
                     f'n_neighbors={config["params"]["n_neighbors"]}, '
                     f'min_dist={config["params"]["min_dist"]}, '
                     f'metric={config["params"]["metric"]}')
        plt.colorbar(scatter)
        plt.tight_layout()
        
        filename_3d = os.path.join(output_dir, f'umap_3d_{config["name"]}.png')
        plt.savefig(filename_3d, dpi=150, bbox_inches='tight')
        print(f"    Saved: {filename_3d}")
        plt.close()

def main():
    # Option to choose data source and content type
    use_db = True  # Set to True to use database, False to use JSON files
    use_content_snippets = True  # Set to False to use document titles instead of content snippets

    if use_db:
        # Load data from database
        print("Loading data from database...")
        embeddings, content_or_titles, file_paths = load_data_from_db(use_content_snippets=use_content_snippets)
    else:
        # Load data from JSON files
        print("Loading data from JSON files...")
        embeddings, content_or_titles, file_paths = load_data()

    print(f"\nData loaded:")
    print(f"  - Embedding shape: {embeddings.shape}")
    print(f"  - Number of documents: {len(content_or_titles)}")
    print(f"  - Content type: {'Content snippets' if use_content_snippets else 'Document titles'}")

    # Standardize embeddings
    standardized_embeddings, scaler = standardize_embeddings(embeddings)

    # Create plots with different UMAP settings
    create_umap_plots(standardized_embeddings, file_paths, {})

    print("\nVisualization complete! Plots saved to 'plots' directory.")

if __name__ == "__main__":
    main()