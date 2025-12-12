"""
Vault Organizer - Main Orchestrator

This script analyzes a Markdown vault to identify thematically inconsistent
folders and suggests file relocations based on semantic similarity.

Usage:
    python vault_organizer.py analyze [OPTIONS]
    python vault_organizer.py preview [OPTIONS]
"""

import json
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

from folder_tree import (
    build_tree, 
    compute_folder_embeddings, 
    get_all_folders,
    get_all_files,
    save_folder_embeddings,
    load_folder_embeddings,
    folder_embeddings_to_dict,
    print_tree
)
from discrepancy import (
    rank_incoherent_folders,
    identify_all_outliers,
    print_folder_analysis,
    print_outliers
)
from suggestions import (
    generate_suggestions,
    print_suggestions,
    generate_move_commands,
    AnalysisReport
)
from populate_sqlite_vec_db import init_sqlite_vec, deserialize_f32


# --- Data Loading ---

def load_embeddings_from_json(path: str = "data/doc_emb.json") -> dict[str, np.ndarray]:
    """Load pre-computed embeddings from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    result = {}
    for file_path, emb in data.items():
        result[file_path] = np.array(emb, dtype=np.float32)
    
    print(f"✅ Loaded {len(result)} file embeddings from {path}")
    return result


def load_embeddings_from_db(db_path: str = "data/db.db") -> dict[str, np.ndarray]:
    """Load embeddings from SQLite database."""
    with init_sqlite_vec(db_path) as conn:
        cursor = conn.execute("SELECT id, document_embedding FROM vec_emb;")
        results = cursor.fetchall()
    
    embeddings = {}
    for file_id, emb_blob in results:
        embeddings[file_id] = np.array(deserialize_f32(emb_blob), dtype=np.float32)
    
    print(f"✅ Loaded {len(embeddings)} file embeddings from {db_path}")
    return embeddings


# --- Main Pipeline ---

def analyze_vault(
    doc_emb_path: str = "data/doc_emb.json",
    dir_emb_path: str = "data/dir_emb.json",
    z_threshold: float = 2.0,
    min_files: int = 3,
    top_k_suggestions: int = 3,
    min_similarity: float = 0.5,
    recompute_folders: bool = False,
    verbose: bool = True
) -> AnalysisReport:
    """
    Run the complete vault analysis pipeline.
    
    Args:
        doc_emb_path: Path to document embeddings JSON
        dir_emb_path: Path to save/load folder embeddings
        z_threshold: Z-score threshold for outlier detection
        min_files: Minimum files in folder to analyze
        top_k_suggestions: Number of relocation candidates per file
        min_similarity: Minimum similarity for candidate folders
        recompute_folders: Force recomputation of folder embeddings
        verbose: Print progress and results
    
    Returns:
        AnalysisReport with all findings
    """
    # Step 1: Load file embeddings
    if verbose:
        print("\n📚 Step 1: Loading file embeddings...")
    
    file_embeddings = load_embeddings_from_json(doc_emb_path)
    
    # Step 2: Build folder tree
    if verbose:
        print("\n🌳 Step 2: Building folder tree...")
    
    root = build_tree(file_embeddings)
    all_folders = get_all_folders(root)
    all_files = get_all_files(root)
    
    if verbose:
        print(f"   Found {len(all_folders)} folders and {len(all_files)} files")
    
    # Step 3: Compute or load folder embeddings
    dir_emb_file = Path(dir_emb_path)
    
    if not recompute_folders and dir_emb_file.exists():
        if verbose:
            print(f"\n📂 Step 3: Loading cached folder embeddings from {dir_emb_path}...")
        folder_embeddings = load_folder_embeddings(dir_emb_path)
        
        # Assign embeddings back to tree nodes
        for folder in all_folders:
            if folder.path in folder_embeddings:
                folder.embedding = folder_embeddings[folder.path]
    else:
        if verbose:
            print("\n🔄 Step 3: Computing folder embeddings (bottom-up aggregation)...")
        compute_folder_embeddings(root)
        save_folder_embeddings(root, dir_emb_path)
        folder_embeddings = folder_embeddings_to_dict(root)
    
    # Convert to numpy arrays if needed
    folder_emb_np = {
        k: (v if isinstance(v, np.ndarray) else np.array(v, dtype=np.float32))
        for k, v in folder_embeddings.items()
    }
    
    if verbose:
        print(f"   Computed embeddings for {len(folder_emb_np)} folders")
    
    # Step 4: Analyze folder coherence
    if verbose:
        print("\n📊 Step 4: Analyzing folder coherence...")
    
    folder_analyses = rank_incoherent_folders(root, min_files=min_files)
    
    if verbose:
        print_folder_analysis(folder_analyses, top_n=10)
    
    # Step 5: Identify outlier files
    if verbose:
        print("\n🔍 Step 5: Identifying outlier files...")
    
    outliers = identify_all_outliers(root, z_threshold=z_threshold, min_files=min_files)
    
    if verbose:
        print(f"   Found {len(outliers)} outlier files (z > {z_threshold})")
        print_outliers(outliers, top_n=15)
    
    # Step 6: Generate relocation suggestions
    if verbose:
        print("\n💡 Step 6: Generating relocation suggestions...")
    
    suggestions = generate_suggestions(
        outliers, 
        folder_emb_np,
        k=top_k_suggestions,
        min_similarity=min_similarity
    )
    
    if verbose:
        print_suggestions(suggestions, top_n=15)
    
    # Create report
    report = AnalysisReport(
        analysis_date=datetime.now().isoformat(),
        total_files=len(all_files),
        total_folders=len(all_folders),
        outlier_count=len(outliers),
        z_threshold=z_threshold,
        suggestions=suggestions
    )
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Vault Organizer - Semantic file organization tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze vault and generate suggestions")
    analyze_parser.add_argument(
        "--doc-emb", 
        default="data/doc_emb.json",
        help="Path to document embeddings JSON"
    )
    analyze_parser.add_argument(
        "--dir-emb",
        default="data/dir_emb.json", 
        help="Path to folder embeddings JSON"
    )
    analyze_parser.add_argument(
        "--output", "-o",
        default="data/suggestions.json",
        help="Output path for suggestions (use '-' for stdout)"
    )
    analyze_parser.add_argument(
        "--z-threshold", "-z",
        type=float,
        default=2.0,
        help="Z-score threshold for outlier detection (default: 2.0)"
    )
    analyze_parser.add_argument(
        "--min-files",
        type=int,
        default=3,
        help="Minimum files in folder to analyze (default: 3)"
    )
    analyze_parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=3,
        help="Number of suggestion candidates per file (default: 3)"
    )
    analyze_parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.5,
        help="Minimum similarity for candidate folders (default: 0.5)"
    )
    analyze_parser.add_argument(
        "--recompute",
        action="store_true",
        help="Force recomputation of folder embeddings"
    )
    analyze_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    
    # Preview command
    preview_parser = subparsers.add_parser("preview", help="Preview folder structure")
    preview_parser.add_argument(
        "--doc-emb",
        default="data/doc_emb.json",
        help="Path to document embeddings JSON"
    )
    preview_parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum depth to display (default: 3)"
    )
    
    # Generate move commands
    moves_parser = subparsers.add_parser("moves", help="Generate move commands from suggestions")
    moves_parser.add_argument(
        "--suggestions",
        default="data/suggestions.json",
        help="Path to suggestions JSON file"
    )
    moves_parser.add_argument(
        "--vault-root",
        default="",
        help="Root path of the vault to prefix commands"
    )
    
    args = parser.parse_args()
    
    if args.command == "analyze":
        report = analyze_vault(
            doc_emb_path=args.doc_emb,
            dir_emb_path=args.dir_emb,
            z_threshold=args.z_threshold,
            min_files=args.min_files,
            top_k_suggestions=args.top_k,
            min_similarity=args.min_similarity,
            recompute_folders=args.recompute,
            verbose=not args.quiet
        )
        
        if args.output == "-":
            print(json.dumps(report.to_dict(), indent=2))
        else:
            report.save(args.output)
            print(f"\n✨ Analysis complete! {len(report.suggestions)} suggestions saved to {args.output}")
    
    elif args.command == "preview":
        file_embeddings = load_embeddings_from_json(args.doc_emb)
        root = build_tree(file_embeddings)
        
        def limited_print(folder, depth=0, max_depth=3):
            if depth > max_depth:
                return
            prefix = "  " * depth
            name = folder.name or "<root>"
            print(f"{prefix}📁 {name} ({len(folder.files)} files)")
            for sub in folder.subfolders:
                limited_print(sub, depth + 1, max_depth)
        
        limited_print(root, max_depth=args.max_depth)
    
    elif args.command == "moves":
        with open(args.suggestions, 'r') as f:
            data = json.load(f)
        
        # Reconstruct Suggestion objects
        from suggestions import Suggestion, RelocationCandidate
        suggestions = []
        for s in data.get("suggestions", []):
            candidates = [
                RelocationCandidate(c["folder"], c["similarity"])
                for c in s.get("candidates", [])
            ]
            suggestions.append(Suggestion(
                file_path=s["file_path"],
                current_folder=s["current_folder"],
                deviation_score=s["deviation_score"],
                z_score=s["z_score"],
                candidates=candidates
            ))
        
        commands = generate_move_commands(suggestions, args.vault_root)
        
        print("# Generated move commands")
        print("# Review carefully before executing!\n")
        for cmd in commands:
            print(cmd)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
