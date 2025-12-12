"""
Relocation Suggestions

This module generates suggestions for moving outlier files to more
semantically appropriate folders based on embedding similarity.
"""

import json
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import PurePosixPath
from folder_tree import FolderNode, FileNode, get_all_folders
from discrepancy import FileOutlier, cosine_similarity


@dataclass
class RelocationCandidate:
    """A potential destination folder for a file."""
    folder_path: str
    similarity: float


@dataclass
class Suggestion:
    """A complete relocation suggestion for an outlier file."""
    file_path: str
    current_folder: str
    deviation_score: float
    z_score: float
    candidates: list[RelocationCandidate]
    
    def to_dict(self):
        return {
            "file_path": self.file_path,
            "current_folder": self.current_folder,
            "deviation_score": round(self.deviation_score, 4),
            "z_score": round(self.z_score, 2),
            "candidates": [
                {"folder": c.folder_path, "similarity": round(c.similarity, 4)}
                for c in self.candidates
            ]
        }


def get_ancestor_paths(path: str) -> set[str]:
    """
    Get all ancestor folder paths for a given path.
    
    Example: "a/b/c/file.md" -> {"a", "a/b", "a/b/c"}
    """
    ancestors = set()
    p = PurePosixPath(path).parent
    
    while str(p) and str(p) != ".":
        ancestors.add(str(p))
        p = p.parent
    
    return ancestors


def find_similar_folders(
    file: FileNode,
    folder_embeddings: dict[str, np.ndarray],
    k: int = 5,
    exclude_ancestors: bool = True
) -> list[RelocationCandidate]:
    """
    Find top-K folders most similar to a file's embedding.
    
    Args:
        file: FileNode with embedding
        folder_embeddings: Dict mapping folder paths to embeddings
        k: Number of suggestions to return
        exclude_ancestors: Whether to exclude the file's current folder hierarchy
    
    Returns:
        List of RelocationCandidate objects, sorted by similarity descending
    """
    if file.embedding is None:
        return []
    
    # Get paths to exclude (current folder and its ancestors)
    exclude_paths = set()
    if exclude_ancestors:
        current_folder = str(PurePosixPath(file.path).parent)
        if current_folder != ".":
            exclude_paths.add(current_folder)
            exclude_paths.update(get_ancestor_paths(file.path))
    
    # Compute similarities
    candidates = []
    for folder_path, folder_emb in folder_embeddings.items():
        if folder_path in exclude_paths:
            continue
        
        sim = cosine_similarity(file.embedding, folder_emb)
        candidates.append(RelocationCandidate(folder_path=folder_path, similarity=sim))
    
    # Sort by similarity descending and take top K
    candidates.sort(key=lambda x: x.similarity, reverse=True)
    return candidates[:k]


def generate_suggestions(
    outliers: list[FileOutlier],
    folder_embeddings: dict[str, np.ndarray],
    k: int = 3,
    min_similarity: float = 0.5
) -> list[Suggestion]:
    """
    Generate relocation suggestions for a list of outlier files.
    
    Args:
        outliers: List of FileOutlier objects
        folder_embeddings: Dict mapping folder paths to embeddings
        k: Number of candidate folders per file
        min_similarity: Minimum similarity threshold for candidates
    
    Returns:
        List of Suggestion objects
    """
    suggestions = []
    
    for outlier in outliers:
        candidates = find_similar_folders(
            outlier.file, 
            folder_embeddings, 
            k=k * 2  # Get extra to filter by threshold
        )
        
        # Filter by minimum similarity
        candidates = [c for c in candidates if c.similarity >= min_similarity][:k]
        
        if candidates:  # Only suggest if we found good candidates
            suggestion = Suggestion(
                file_path=outlier.file.path,
                current_folder=outlier.folder.path,
                deviation_score=outlier.deviation,
                z_score=outlier.z_score,
                candidates=candidates
            )
            suggestions.append(suggestion)
    
    return suggestions


@dataclass
class AnalysisReport:
    """Complete analysis report."""
    analysis_date: str
    total_files: int
    total_folders: int
    outlier_count: int
    z_threshold: float
    suggestions: list[Suggestion]
    
    def to_dict(self):
        return {
            "analysis_date": self.analysis_date,
            "total_files": self.total_files,
            "total_folders": self.total_folders,
            "outlier_count": self.outlier_count,
            "z_threshold": self.z_threshold,
            "suggestions": [s.to_dict() for s in self.suggestions]
        }
    
    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"✅ Saved report to {path}")


def print_suggestions(suggestions: list[Suggestion], top_n: int = 10) -> None:
    """Print suggestions in a human-readable format."""
    print(f"\n{'='*70}")
    print(f"📋 Relocation Suggestions (showing top {min(top_n, len(suggestions))})")
    print(f"{'='*70}\n")
    
    for i, s in enumerate(suggestions[:top_n], 1):
        print(f"{i}. 📄 {s.file_path}")
        print(f"   Current: {s.current_folder}")
        print(f"   Deviation: {s.deviation_score:.3f} | Z-Score: {s.z_score:.2f}")
        print(f"   Suggested destinations:")
        for j, c in enumerate(s.candidates, 1):
            print(f"     {j}. 📁 {c.folder_path} (similarity: {c.similarity:.3f})")
        print()


def generate_move_commands(suggestions: list[Suggestion], vault_root: str = "") -> list[str]:
    """
    Generate shell commands to execute the moves.
    
    Uses the top suggestion for each file.
    
    Args:
        suggestions: List of Suggestion objects
        vault_root: Root path of the vault (prefix for paths)
    
    Returns:
        List of mv commands as strings
    """
    commands = []
    
    for s in suggestions:
        if not s.candidates:
            continue
        
        best_candidate = s.candidates[0]
        
        if vault_root:
            source = f"{vault_root}/{s.file_path}"
            dest = f"{vault_root}/{best_candidate.folder_path}/"
        else:
            source = s.file_path
            dest = f"{best_candidate.folder_path}/"
        
        commands.append(f'mv "{source}" "{dest}"')
    
    return commands
