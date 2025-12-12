"""
Inbox Organizer - Main Orchestrator

Organizes files from an unorganized working directory by:
1. Clustering similar files together
2. Finding suitable destination folders for each cluster

Usage:
    python inbox_organizer.py --inbox "FuckHere" --threshold 0.3
"""

import json
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path, PurePosixPath
from dataclasses import dataclass, asdict

from clustering import (
    load_inbox_files,
    cluster_files,
    label_all_clusters,
    print_clusters,
    get_cluster_stats,
    FileCluster,
    cosine_similarity
)
from folder_tree import (
    build_tree,
    compute_folder_embeddings,
    get_all_folders,
    save_folder_embeddings,
    load_folder_embeddings,
    folder_embeddings_to_dict,
    FolderNode
)


# --- Data Loading ---

def load_embeddings_from_json(path: str = "data/doc_emb.json") -> dict[str, np.ndarray]:
    """Load pre-computed embeddings from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    result = {}
    for file_path, emb in data.items():
        result[file_path] = np.array(emb, dtype=np.float32)
    
    return result


# --- Folder Matching ---

def get_ancestor_paths(path: str) -> set[str]:
    """Get all ancestor folder paths for a given path."""
    ancestors = set()
    p = PurePosixPath(path)
    
    while str(p.parent) and str(p.parent) != "." and str(p.parent) != str(p):
        ancestors.add(str(p.parent))
        p = p.parent
    
    return ancestors


def find_matching_folders(
    cluster: FileCluster,
    folder_embeddings: dict[str, np.ndarray],
    inbox_prefix: str,
    k: int = 5,
    min_similarity: float = 0.0
) -> list[tuple[str, float]]:
    """
    Find top-K folders most similar to a cluster's centroid.
    
    Args:
        cluster: FileCluster with computed centroid
        folder_embeddings: Dict mapping folder paths to embeddings
        inbox_prefix: Path prefix to exclude from candidates
        k: Number of suggestions to return
        min_similarity: Minimum similarity threshold
    
    Returns:
        List of (folder_path, similarity) tuples, sorted by similarity descending
    """
    if cluster.centroid is None:
        return []
    
    # Normalize inbox prefix
    inbox_prefix = inbox_prefix.rstrip('/')
    
    candidates = []
    for folder_path, folder_emb in folder_embeddings.items():
        # Skip folders within the inbox
        if folder_path.startswith(inbox_prefix + '/') or folder_path == inbox_prefix:
            continue
        if '/' + inbox_prefix + '/' in '/' + folder_path + '/':
            continue
        
        # Skip empty path (root)
        if not folder_path or folder_path == '.':
            continue
        
        sim = cosine_similarity(cluster.centroid, folder_emb)
        if sim >= min_similarity:
            candidates.append((folder_path, sim))
    
    # Sort by similarity descending
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    return candidates[:k]


# --- Suggestion Data Structures ---

@dataclass
class ClusterSuggestion:
    """Suggestion for where to move a cluster of files."""
    cluster_id: int
    files: list[str]
    file_count: int
    label: str | None
    coherence: float
    candidates: list[tuple[str, float]]
    
    def to_dict(self):
        return {
            "cluster_id": self.cluster_id,
            "file_count": self.file_count,
            "label": self.label,
            "coherence": round(self.coherence, 4),
            "files": self.files,
            "suggested_destinations": [
                {"folder": path, "similarity": round(sim, 4)}
                for path, sim in self.candidates
            ]
        }


@dataclass
class InboxReport:
    """Complete analysis report for inbox organization."""
    analysis_date: str
    inbox_path: str
    total_files: int
    cluster_count: int
    distance_threshold: float
    suggestions: list[ClusterSuggestion]
    
    def to_dict(self):
        return {
            "analysis_date": self.analysis_date,
            "inbox_path": self.inbox_path,
            "total_files": self.total_files,
            "cluster_count": self.cluster_count,
            "distance_threshold": self.distance_threshold,
            "clusters": [s.to_dict() for s in self.suggestions]
        }
    
    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"✅ Saved report to {path}")


# --- Main Pipeline ---

def organize_inbox(
    inbox_prefix: str,
    doc_emb_path: str = "data/doc_emb.json",
    dir_emb_path: str = "data/dir_emb.json",
    distance_threshold: float = 0.3,
    top_k: int = 3,
    min_similarity: float = 0.5,
    recompute_folders: bool = False,
    verbose: bool = True
) -> InboxReport:
    """
    Analyze inbox folder and generate organization suggestions.
    
    Args:
        inbox_prefix: Path prefix for the working directory (e.g., "FuckHere")
        doc_emb_path: Path to document embeddings JSON
        dir_emb_path: Path to folder embeddings JSON
        distance_threshold: Clustering distance threshold (lower = more clusters)
        top_k: Number of destination suggestions per cluster
        min_similarity: Minimum folder similarity to include
        recompute_folders: Force recomputation of folder embeddings
        verbose: Print progress and results
    
    Returns:
        InboxReport with all findings
    """
    inbox_prefix = inbox_prefix.rstrip('/')
    
    # Step 1: Load all embeddings
    if verbose:
        print(f"\n📚 Step 1: Loading embeddings from {doc_emb_path}...")
    
    all_embeddings = load_embeddings_from_json(doc_emb_path)
    if verbose:
        print(f"   Loaded {len(all_embeddings)} file embeddings")
    
    # Step 2: Extract inbox files
    if verbose:
        print(f"\n📥 Step 2: Extracting files from '{inbox_prefix}'...")
    
    inbox_embeddings = load_inbox_files(inbox_prefix, all_embeddings)
    
    if not inbox_embeddings:
        print(f"❌ No files found matching prefix '{inbox_prefix}'")
        print(f"   Available top-level paths: {set(p.split('/')[0] for p in all_embeddings.keys())}")
        return InboxReport(
            analysis_date=datetime.now().isoformat(),
            inbox_path=inbox_prefix,
            total_files=0,
            cluster_count=0,
            distance_threshold=distance_threshold,
            suggestions=[]
        )
    
    if verbose:
        print(f"   Found {len(inbox_embeddings)} files in inbox")
    
    # Step 3: Cluster inbox files
    if verbose:
        print(f"\n🔗 Step 3: Clustering files (threshold={distance_threshold})...")
    
    clusters = cluster_files(inbox_embeddings, distance_threshold=distance_threshold)
    label_all_clusters(clusters)
    
    if verbose:
        stats = get_cluster_stats(clusters)
        print(f"   Created {stats['total_clusters']} clusters")
        print(f"   Largest: {stats['largest_cluster']} files | Singletons: {stats['singletons']}")
        print_clusters(clusters, show_files=True, max_files=3)
    
    # Step 4: Load or compute folder embeddings (excluding inbox)
    if verbose:
        print(f"\n📂 Step 4: Loading folder embeddings...")
    
    dir_emb_file = Path(dir_emb_path)
    
    if not recompute_folders and dir_emb_file.exists():
        folder_embeddings = load_folder_embeddings(dir_emb_path)
        if verbose:
            print(f"   Loaded {len(folder_embeddings)} folder embeddings from cache")
    else:
        if verbose:
            print(f"   Computing folder embeddings (excluding inbox)...")
        
        # Build tree excluding inbox files for folder embedding computation
        non_inbox_embeddings = {
            k: v for k, v in all_embeddings.items()
            if not k.startswith(inbox_prefix + '/')
        }
        
        root = build_tree(non_inbox_embeddings)
        compute_folder_embeddings(root)
        save_folder_embeddings(root, dir_emb_path)
        folder_embeddings = folder_embeddings_to_dict(root)
    
    # Convert to numpy arrays
    folder_emb_np = {
        k: (v if isinstance(v, np.ndarray) else np.array(v, dtype=np.float32))
        for k, v in folder_embeddings.items()
    }
    
    # Step 5: Match clusters to destination folders
    if verbose:
        print(f"\n🎯 Step 5: Finding destination folders for each cluster...")
    
    suggestions = []
    for cluster in clusters:
        candidates = find_matching_folders(
            cluster,
            folder_emb_np,
            inbox_prefix,
            k=top_k,
            min_similarity=min_similarity
        )
        
        suggestions.append(ClusterSuggestion(
            cluster_id=cluster.cluster_id,
            files=cluster.files,
            file_count=len(cluster.files),
            label=cluster.label,
            coherence=cluster.coherence,
            candidates=candidates
        ))
    
    # Print suggestions
    if verbose:
        print_suggestions(suggestions)
    
    # Create report
    report = InboxReport(
        analysis_date=datetime.now().isoformat(),
        inbox_path=inbox_prefix,
        total_files=len(inbox_embeddings),
        cluster_count=len(clusters),
        distance_threshold=distance_threshold,
        suggestions=suggestions
    )
    
    return report


def print_suggestions(suggestions: list[ClusterSuggestion]) -> None:
    """Print suggestions in a human-readable format."""
    print(f"\n{'═'*70}")
    print(f"📋 ORGANIZATION SUGGESTIONS")
    print(f"{'═'*70}\n")
    
    for s in suggestions:
        label = s.label or f"Cluster {s.cluster_id}"
        print(f"┌{'─'*68}┐")
        print(f"│ Cluster #{s.cluster_id}: \"{label}\" ({s.file_count} files)")
        print(f"│ Coherence: {s.coherence:.3f}")
        print(f"├{'─'*68}┤")
        
        # Show files (abbreviated)
        for f in s.files[:3]:
            filename = f.rsplit('/', 1)[-1]
            print(f"│   📄 {filename}")
        if len(s.files) > 3:
            print(f"│   ... +{len(s.files) - 3} more")
        
        print(f"│")
        print(f"│ ➡️  Suggested destinations:")
        if s.candidates:
            for i, (folder, sim) in enumerate(s.candidates, 1):
                print(f"│     {i}. 📁 {folder} ({sim:.3f})")
        else:
            print(f"│     ⚠️  No suitable folders found (try lowering --min-similarity)")
        
        print(f"└{'─'*68}┘")
        print()


def generate_move_commands(suggestions: list[ClusterSuggestion], vault_root: str = "") -> list[str]:
    """Generate shell commands to move files to their suggested destinations."""
    commands = []
    
    for s in suggestions:
        if not s.candidates:
            continue
        
        best_folder = s.candidates[0][0]
        
        # Add comment for this cluster
        commands.append(f"\n# Cluster {s.cluster_id}: {s.label} ({s.file_count} files)")
        
        for file_path in s.files:
            if vault_root:
                source = f"{vault_root}/{file_path}"
                dest = f"{vault_root}/{best_folder}/"
            else:
                source = file_path
                dest = f"{best_folder}/"
            
            commands.append(f'mv "{source}" "{dest}"')
    
    return commands


# --- CLI ---

def main():
    parser = argparse.ArgumentParser(
        description="Inbox Organizer - Organize files from a working directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inbox_organizer.py --inbox "FuckHere"
  python inbox_organizer.py --inbox "Notes/FuckHere" --threshold 0.4 --top-k 5
  python inbox_organizer.py --inbox "FuckHere" --output inbox_suggestions.json
        """
    )
    
    parser.add_argument(
        "--inbox", "-i",
        required=True,
        help="Path prefix for the working directory (e.g., 'FuckHere' or 'Notes/FuckHere')"
    )
    parser.add_argument(
        "--doc-emb",
        default="data/doc_emb.json",
        help="Path to document embeddings JSON (default: data/doc_emb.json)"
    )
    parser.add_argument(
        "--dir-emb",
        default="data/dir_emb.json",
        help="Path to folder embeddings JSON (default: data/dir_emb.json)"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/inbox_suggestions.json",
        help="Output path for suggestions (use '-' for stdout, default: data/inbox_suggestions.json)"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.3,
        help="Clustering distance threshold, lower = more clusters (default: 0.3, range: 0.1-0.6)"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=3,
        help="Number of destination suggestions per cluster (default: 3)"
    )
    parser.add_argument(
        "--min-similarity", "-s",
        type=float,
        default=0.5,
        help="Minimum similarity for candidate folders (default: 0.5)"
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Force recomputation of folder embeddings"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    parser.add_argument(
        "--moves",
        action="store_true",
        help="Output move commands instead of JSON report"
    )
    parser.add_argument(
        "--vault-root",
        default="",
        help="Root path of the vault for move commands"
    )
    
    args = parser.parse_args()
    
    report = organize_inbox(
        inbox_prefix=args.inbox,
        doc_emb_path=args.doc_emb,
        dir_emb_path=args.dir_emb,
        distance_threshold=args.threshold,
        top_k=args.top_k,
        min_similarity=args.min_similarity,
        recompute_folders=args.recompute,
        verbose=not args.quiet
    )
    
    if args.moves:
        # Output move commands
        commands = generate_move_commands(report.suggestions, args.vault_root)
        print("\n# Generated move commands")
        print("# Review carefully before executing!\n")
        for cmd in commands:
            print(cmd)
    elif args.output == "-":
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        report.save(args.output)
        print(f"\n✨ Analysis complete!")
        print(f"   Files in inbox: {report.total_files}")
        print(f"   Clusters found: {report.cluster_count}")
        print(f"   Suggestions saved to: {args.output}")


if __name__ == "__main__":
    main()
