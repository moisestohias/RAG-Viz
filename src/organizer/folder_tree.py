"""
Folder Tree Construction & Bottom-Up Embedding Aggregation

This module builds a hierarchical tree structure from file paths and computes
folder-level embeddings by aggregating child embeddings bottom-up.
"""

import json
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import PurePosixPath


@dataclass
class FileNode:
    """Represents a file in the vault with its embedding."""
    path: str                    # Relative path (e.g., "Notes/Python/async.md")
    embedding: np.ndarray        # 1024-dim vector
    parent: "FolderNode | None" = None
    
    @property
    def name(self):
        return PurePosixPath(self.path).name
    
    @property
    def parent_path(self):
        return str(PurePosixPath(self.path).parent)


@dataclass
class FolderNode:
    """Represents a folder in the vault with aggregated embedding."""
    path: str                              # Folder path (e.g., "Notes/Python")
    files: list[FileNode] = field(default_factory=list)
    subfolders: list["FolderNode"] = field(default_factory=list)
    embedding: np.ndarray | None = None    # Computed bottom-up
    parent: "FolderNode | None" = None
    
    @property
    def name(self):
        return PurePosixPath(self.path).name if self.path else "<root>"
    
    @property
    def is_leaf(self):
        """True if folder has no subfolders."""
        return len(self.subfolders) == 0
    
    @property
    def total_files(self):
        """Count all files recursively."""
        count = len(self.files)
        for sub in self.subfolders:
            count += sub.total_files
        return count


def build_tree(file_embeddings: dict[str, np.ndarray]) -> FolderNode:
    """
    Build a hierarchical folder tree from a flat dict of file paths to embeddings.
    
    Args:
        file_embeddings: Dict mapping file paths to their embedding vectors
                         e.g., {"Notes/Python/async.md": np.array([...])}
    
    Returns:
        Root FolderNode containing the entire tree structure
    """
    # Create root node
    root = FolderNode(path="")
    
    # Cache for folder nodes by path
    folder_cache: dict[str, FolderNode] = {"": root}
    
    def get_or_create_folder(folder_path: str) -> FolderNode:
        """Recursively get or create a folder and its parents."""
        if folder_path in folder_cache:
            return folder_cache[folder_path]
        
        # Get parent path
        path_obj = PurePosixPath(folder_path)
        parent_path = str(path_obj.parent) if path_obj.parent != path_obj else ""
        
        # Ensure parent exists
        parent_folder = get_or_create_folder(parent_path)
        
        # Create this folder
        new_folder = FolderNode(path=folder_path, parent=parent_folder)
        parent_folder.subfolders.append(new_folder)
        folder_cache[folder_path] = new_folder
        
        return new_folder
    
    # Process all files
    for file_path, embedding in file_embeddings.items():
        # Ensure embedding is numpy array
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype=np.float32)
        
        # Get parent folder path
        path_obj = PurePosixPath(file_path)
        parent_path = str(path_obj.parent) if str(path_obj.parent) != "." else ""
        
        # Get or create parent folder
        parent_folder = get_or_create_folder(parent_path)
        
        # Create file node
        file_node = FileNode(path=file_path, embedding=embedding, parent=parent_folder)
        parent_folder.files.append(file_node)
    
    return root


def compute_folder_embeddings(folder: FolderNode) -> None:
    """
    Recursively compute folder embeddings bottom-up.
    
    Each folder's embedding is the normalized mean of all its descendants' embeddings.
    Must process children before parents to ensure proper aggregation.
    
    Args:
        folder: Folder node to process (typically root)
    """
    # 1. Recursively compute child folder embeddings first (bottom-up)
    for subfolder in folder.subfolders:
        compute_folder_embeddings(subfolder)
    
    # 2. Collect all descendant embeddings
    all_embeddings = []
    
    # Add direct file embeddings
    for file in folder.files:
        if file.embedding is not None:
            all_embeddings.append(file.embedding)
    
    # Add subfolder embeddings (already computed)
    for subfolder in folder.subfolders:
        if subfolder.embedding is not None:
            all_embeddings.append(subfolder.embedding)
    
    # 3. Compute folder embedding as normalized mean
    if all_embeddings:
        folder.embedding = np.mean(all_embeddings, axis=0)
        norm = np.linalg.norm(folder.embedding)
        if norm > 0:
            folder.embedding = folder.embedding / norm


def get_all_folders(root: FolderNode, include_root: bool = False) -> list[FolderNode]:
    """
    Flatten tree to a list of all folder nodes.
    
    Args:
        root: Root folder node
        include_root: Whether to include the root node itself
    
    Returns:
        List of all FolderNode objects in the tree
    """
    folders = []
    
    def traverse(node: FolderNode):
        if node.path or include_root:  # Skip empty root unless requested
            folders.append(node)
        for sub in node.subfolders:
            traverse(sub)
    
    traverse(root)
    return folders


def get_all_files(root: FolderNode) -> list[FileNode]:
    """
    Flatten tree to a list of all file nodes.
    
    Args:
        root: Root folder node
    
    Returns:
        List of all FileNode objects in the tree
    """
    files = []
    
    def traverse(node: FolderNode):
        files.extend(node.files)
        for sub in node.subfolders:
            traverse(sub)
    
    traverse(root)
    return files


def print_tree(folder: FolderNode, indent: int = 0) -> None:
    """
    Print tree structure for debugging.
    """
    prefix = "  " * indent
    name = folder.name or "<root>"
    has_emb = "✓" if folder.embedding is not None else "✗"
    print(f"{prefix}📁 {name} ({len(folder.files)} files, {len(folder.subfolders)} subfolders) [{has_emb}]")
    
    for file in folder.files:
        print(f"{prefix}  📄 {file.name}")
    
    for sub in folder.subfolders:
        print_tree(sub, indent + 1)


# --- Serialization ---

def folder_embeddings_to_dict(root: FolderNode) -> dict[str, list[float]]:
    """
    Extract all folder embeddings as a serializable dict.
    
    Returns:
        Dict mapping folder paths to embedding lists
    """
    result = {}
    for folder in get_all_folders(root):
        if folder.embedding is not None:
            result[folder.path] = folder.embedding.tolist()
    return result


def save_folder_embeddings(root: FolderNode, path: str = "data/dir_emb.json") -> None:
    """Save folder embeddings to JSON file."""
    data = folder_embeddings_to_dict(root)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved {len(data)} folder embeddings to {path}")


def load_folder_embeddings(path: str = "data/dir_emb.json") -> dict[str, np.ndarray]:
    """Load folder embeddings from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {k: np.array(v, dtype=np.float32) for k, v in data.items()}
