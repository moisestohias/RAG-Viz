"""
Semantic Discrepancy Detection

This module analyzes folder coherence and identifies outlier files
whose embeddings diverge significantly from their folder's semantic center.
"""

import numpy as np
from dataclasses import dataclass
from folder_tree import FolderNode, FileNode, get_all_folders


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Returns value in range [-1, 1], where 1 means identical direction.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_file_deviations(folder: FolderNode) -> dict[str, float]:
    """
    For each direct child file, compute its deviation from the folder centroid.
    
    Deviation = 1 - cosine_similarity(file_emb, folder_emb)
    Higher values indicate files that are semantically distant from the folder theme.
    
    Args:
        folder: FolderNode with computed embedding
    
    Returns:
        Dict mapping file paths to deviation scores [0, 2]
    """
    if folder.embedding is None:
        return {}
    
    deviations = {}
    for file in folder.files:
        if file.embedding is not None:
            sim = cosine_similarity(file.embedding, folder.embedding)
            deviations[file.path] = 1.0 - sim
    
    return deviations


def compute_folder_coherence(folder: FolderNode) -> float:
    """
    Compute folder coherence as mean cosine similarity of files to centroid.
    
    Higher values indicate more thematically consistent folders.
    
    Args:
        folder: FolderNode with computed embedding
    
    Returns:
        Coherence score in range [0, 1], or -1 if cannot compute
    """
    if folder.embedding is None or not folder.files:
        return -1.0
    
    similarities = []
    for file in folder.files:
        if file.embedding is not None:
            sim = cosine_similarity(file.embedding, folder.embedding)
            similarities.append(sim)
    
    if not similarities:
        return -1.0
    
    return float(np.mean(similarities))


def compute_folder_variance(folder: FolderNode) -> float:
    """
    Compute variance of file similarities to folder centroid.
    
    Lower values indicate more consistent folders.
    
    Args:
        folder: FolderNode with computed embedding
    
    Returns:
        Variance score, or -1 if cannot compute
    """
    if folder.embedding is None or not folder.files:
        return -1.0
    
    similarities = []
    for file in folder.files:
        if file.embedding is not None:
            sim = cosine_similarity(file.embedding, folder.embedding)
            similarities.append(sim)
    
    if len(similarities) < 2:
        return 0.0
    
    return float(np.std(similarities))


@dataclass
class FolderAnalysis:
    """Analysis results for a folder."""
    folder: FolderNode
    coherence: float          # Mean similarity to centroid
    variance: float           # Std of similarities
    file_count: int
    outlier_count: int = 0


@dataclass
class FileOutlier:
    """A file identified as an outlier in its folder."""
    file: FileNode
    folder: FolderNode
    deviation: float          # 1 - cosine_sim
    z_score: float            # How many std devs from mean


def rank_incoherent_folders(
    root: FolderNode, 
    min_files: int = 2
) -> list[FolderAnalysis]:
    """
    Rank all folders by incoherence (ascending coherence).
    
    Args:
        root: Root of the folder tree
        min_files: Minimum files required to analyze a folder
    
    Returns:
        List of FolderAnalysis objects, sorted by coherence (lowest first)
    """
    results = []
    
    for folder in get_all_folders(root):
        if len(folder.files) < min_files:
            continue
        
        coherence = compute_folder_coherence(folder)
        variance = compute_folder_variance(folder)
        
        if coherence < 0:
            continue
        
        results.append(FolderAnalysis(
            folder=folder,
            coherence=coherence,
            variance=variance,
            file_count=len(folder.files)
        ))
    
    # Sort by coherence ascending (most incoherent first)
    results.sort(key=lambda x: x.coherence)
    
    return results


def identify_outlier_files(
    folder: FolderNode, 
    z_threshold: float = 2.0
) -> list[FileOutlier]:
    """
    Identify files that are statistical outliers in their folder.
    
    Uses z-score to detect files whose deviation from the folder centroid
    is significantly higher than the folder's average deviation.
    
    Args:
        folder: FolderNode to analyze
        z_threshold: Number of standard deviations to consider as outlier
    
    Returns:
        List of FileOutlier objects for files exceeding threshold
    """
    if folder.embedding is None or not folder.files:
        return []
    
    # Compute deviations for all files
    deviations = []
    file_deviation_pairs = []
    
    for file in folder.files:
        if file.embedding is not None:
            dev = 1.0 - cosine_similarity(file.embedding, folder.embedding)
            deviations.append(dev)
            file_deviation_pairs.append((file, dev))
    
    if len(deviations) < 2:
        return []
    
    # Compute statistics
    mean_dev = np.mean(deviations)
    std_dev = np.std(deviations)
    
    if std_dev == 0:
        return []  # All files have same deviation
    
    # Find outliers
    outliers = []
    for file, dev in file_deviation_pairs:
        z_score = (dev - mean_dev) / std_dev
        if z_score > z_threshold:
            outliers.append(FileOutlier(
                file=file,
                folder=folder,
                deviation=dev,
                z_score=z_score
            ))
    
    # Sort by z_score descending (most extreme first)
    outliers.sort(key=lambda x: x.z_score, reverse=True)
    
    return outliers


def identify_all_outliers(
    root: FolderNode,
    z_threshold: float = 2.0,
    min_files: int = 3
) -> list[FileOutlier]:
    """
    Identify outlier files across the entire vault.
    
    Args:
        root: Root of the folder tree
        z_threshold: Z-score threshold for outlier detection
        min_files: Minimum files in folder to analyze
    
    Returns:
        List of all FileOutlier objects, sorted by z_score descending
    """
    all_outliers = []
    
    for folder in get_all_folders(root):
        if len(folder.files) < min_files:
            continue
        
        outliers = identify_outlier_files(folder, z_threshold)
        all_outliers.extend(outliers)
    
    # Sort by z_score descending
    all_outliers.sort(key=lambda x: x.z_score, reverse=True)
    
    return all_outliers


def print_folder_analysis(analyses: list[FolderAnalysis], top_n: int = 10) -> None:
    """Print top N most incoherent folders."""
    print(f"\n{'='*60}")
    print(f"Top {top_n} Most Incoherent Folders")
    print(f"{'='*60}\n")
    
    for i, analysis in enumerate(analyses[:top_n], 1):
        print(f"{i}. 📁 {analysis.folder.path}")
        print(f"   Coherence: {analysis.coherence:.3f} | Variance: {analysis.variance:.3f} | Files: {analysis.file_count}")
        print()


def print_outliers(outliers: list[FileOutlier], top_n: int = 20) -> None:
    """Print top N outlier files."""
    print(f"\n{'='*60}")
    print(f"Top {top_n} Outlier Files")
    print(f"{'='*60}\n")
    
    for i, outlier in enumerate(outliers[:top_n], 1):
        print(f"{i}. 📄 {outlier.file.path}")
        print(f"   Folder: {outlier.folder.path}")
        print(f"   Deviation: {outlier.deviation:.3f} | Z-Score: {outlier.z_score:.2f}")
        print()
