"""
File Clustering

Groups semantically similar files using Agglomerative Clustering.
"""

import numpy as np
from dataclasses import dataclass, field
from sklearn.cluster import AgglomerativeClustering


@dataclass
class FileCluster:
    """A group of semantically similar files."""
    cluster_id: int
    files: list[str]                  # List of file paths
    centroid: np.ndarray | None = None  # Mean embedding of cluster
    coherence: float = 0.0            # Internal similarity score
    label: str | None = None          # Optional human-readable label


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_internal_coherence(files: list[str], embeddings: dict[str, np.ndarray]) -> float:
    """
    Compute internal coherence of a cluster.
    
    Coherence = mean pairwise cosine similarity between all files.
    Higher values indicate more tightly grouped clusters.
    """
    if len(files) < 2:
        return 1.0  # Single file is perfectly coherent with itself
    
    vecs = [embeddings[f] for f in files if f in embeddings]
    if len(vecs) < 2:
        return 1.0
    
    # Compute mean of pairwise similarities
    similarities = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            similarities.append(cosine_similarity(vecs[i], vecs[j]))
    
    return float(np.mean(similarities)) if similarities else 1.0


def load_inbox_files(
    inbox_prefix: str, 
    all_embeddings: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """
    Extract embeddings for files within the inbox directory.
    
    Args:
        inbox_prefix: Path prefix to match (e.g., "FuckHere" or "Notes/FuckHere")
        all_embeddings: Dict of all file embeddings
    
    Returns:
        Dict of embeddings for files matching the prefix
    """
    # Normalize prefix (ensure no trailing slash, handle both with and without leading path)
    inbox_prefix = inbox_prefix.rstrip('/')
    
    inbox_files = {}
    for path, emb in all_embeddings.items():
        # Match if path starts with prefix or contains prefix as a directory component
        if path.startswith(inbox_prefix + '/') or path.startswith(inbox_prefix + '\\'):
            inbox_files[path] = emb
        elif '/' + inbox_prefix + '/' in path or path == inbox_prefix:
            inbox_files[path] = emb
    
    return inbox_files


def cluster_files(
    embeddings: dict[str, np.ndarray],
    distance_threshold: float = 0.3,
    min_cluster_size: int = 1
) -> list[FileCluster]:
    """
    Group similar files using Agglomerative Clustering.
    
    Args:
        embeddings: Dict mapping file paths to embedding vectors
        distance_threshold: Maximum distance to merge clusters (lower = more clusters)
                           Range: 0.0 (identical only) to 2.0 (everything in one cluster)
                           Recommended: 0.2-0.5
        min_cluster_size: Minimum files per cluster (smaller clusters are kept but flagged)
    
    Returns:
        List of FileCluster objects, sorted by size descending
    """
    if not embeddings:
        return []
    
    paths = list(embeddings.keys())
    X = np.array([embeddings[p] for p in paths])
    
    # Handle single file case
    if len(paths) == 1:
        return [FileCluster(
            cluster_id=0,
            files=paths,
            centroid=X[0] / np.linalg.norm(X[0]),
            coherence=1.0
        )]
    
    # Normalize for cosine distance
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    X_normalized = X / norms
    
    # Perform clustering
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric='cosine',
        linkage='average',
        distance_threshold=distance_threshold
    )
    labels = clustering.fit_predict(X_normalized)
    
    # Group files by cluster label
    cluster_files_map: dict[int, list[str]] = {}
    for path, label in zip(paths, labels):
        if label not in cluster_files_map:
            cluster_files_map[label] = []
        cluster_files_map[label].append(path)
    
    # Create FileCluster objects
    clusters = []
    for cluster_id, files in cluster_files_map.items():
        # Compute centroid (normalized mean)
        cluster_embeddings = [embeddings[f] for f in files]
        centroid = np.mean(cluster_embeddings, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        
        # Compute internal coherence
        coherence = compute_internal_coherence(files, embeddings)
        
        clusters.append(FileCluster(
            cluster_id=cluster_id,
            files=files,
            centroid=centroid,
            coherence=coherence
        ))
    
    # Sort by cluster size descending
    clusters.sort(key=lambda c: len(c.files), reverse=True)
    
    # Re-assign cluster IDs after sorting
    for i, cluster in enumerate(clusters):
        cluster.cluster_id = i
    
    return clusters


def auto_label_cluster(
    cluster: FileCluster,
    max_words: int = 4
) -> str:
    """
    Generate a simple label from file names.
    
    Extracts common words from file names in the cluster.
    """
    from collections import Counter
    import re
    
    # Extract words from file names
    all_words = []
    for path in cluster.files:
        # Get filename without extension
        filename = path.rsplit('/', 1)[-1].rsplit('.', 1)[0]
        # Split by common separators
        words = re.split(r'[-_\s]+', filename.lower())
        # Filter out short/common words
        words = [w for w in words if len(w) > 2 and w not in {'the', 'and', 'for', 'with'}]
        all_words.extend(words)
    
    # Find most common words
    word_counts = Counter(all_words)
    common_words = [word for word, _ in word_counts.most_common(max_words)]
    
    if common_words:
        return ' '.join(common_words[:max_words])
    return f"Cluster {cluster.cluster_id}"


def label_all_clusters(clusters: list[FileCluster]) -> None:
    """Add auto-generated labels to all clusters (modifies in place)."""
    for cluster in clusters:
        if cluster.label is None:
            cluster.label = auto_label_cluster(cluster)


def print_clusters(clusters: list[FileCluster], show_files: bool = True, max_files: int = 5) -> None:
    """Print clusters in a human-readable format."""
    print(f"\n{'='*70}")
    print(f"📊 Found {len(clusters)} clusters")
    print(f"{'='*70}\n")
    
    for cluster in clusters:
        label = cluster.label or f"Cluster {cluster.cluster_id}"
        print(f"┌{'─'*68}┐")
        print(f"│ Cluster #{cluster.cluster_id}: \"{label}\"")
        print(f"│ Files: {len(cluster.files)} | Coherence: {cluster.coherence:.3f}")
        print(f"└{'─'*68}┘")
        
        if show_files:
            files_to_show = cluster.files[:max_files]
            for f in files_to_show:
                # Show just filename, not full path
                filename = f.rsplit('/', 1)[-1]
                print(f"    📄 {filename}")
            if len(cluster.files) > max_files:
                print(f"    ... and {len(cluster.files) - max_files} more files")
        print()


def get_cluster_stats(clusters: list[FileCluster]) -> dict:
    """Get summary statistics for clustering results."""
    if not clusters:
        return {"total_clusters": 0, "total_files": 0}
    
    sizes = [len(c.files) for c in clusters]
    coherences = [c.coherence for c in clusters]
    
    return {
        "total_clusters": len(clusters),
        "total_files": sum(sizes),
        "largest_cluster": max(sizes),
        "smallest_cluster": min(sizes),
        "avg_cluster_size": float(np.mean(sizes)),
        "avg_coherence": float(np.mean(coherences)),
        "singletons": sum(1 for s in sizes if s == 1)
    }
