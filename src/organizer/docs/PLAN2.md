# Vault Organizer v2 - Working Directory Organizer

## New Objective

Organize files from an **unorganized working directory** (e.g., `FuckHere`) by:

1. **Clustering similar files** within the working directory
2. **Matching each cluster** to suitable destination folders elsewhere in the vault

This is an **"inbox zero"** approach—treating the working directory as a dump folder that needs to be emptied into proper locations.

---

## Key Difference from v1

| Aspect | v1 (Previous) | v2 (New) |
|--------|---------------|----------|
| **Scope** | Analyze entire vault for outliers | Focus on specific working directory |
| **Detection** | Z-score outlier detection per folder | Cluster similar files together |
| **Suggestions** | Individual file → folder matches | Cluster of files → folder matches |
| **Goal** | Fix misplaced files everywhere | Empty the "inbox" folder |

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                      inbox_organizer.py                            │
│                      (Main Orchestrator)                           │
└────────────────────────────────────────────────────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│   clustering.py │   │  folder_tree.py │   │  suggestions.py │
│                 │   │   (reuse v1)    │   │   (modified)    │
│ • Load inbox    │   │                 │   │                 │
│   files         │   │ • Build tree    │   │ • Match cluster │
│ • Cluster by    │   │ • Compute       │   │   to folders    │
│   similarity    │   │   folder embs   │   │ • Rank by       │
│ • Label groups  │   │   (exclude      │   │   similarity    │
│                 │   │   inbox)        │   │                 │
└─────────────────┘   └─────────────────┘   └─────────────────┘
```

---

## Phase 1: File Clustering

### Goal
Group files in the working directory into semantically coherent clusters.

### File: `clustering.py`

#### Approach: Agglomerative Clustering

Using **Agglomerative Clustering** with cosine distance because:
- No need to specify number of clusters upfront
- Works well with high-dimensional embeddings
- `distance_threshold` parameter controls granularity

#### Data Structures

```python
@dataclass
class FileCluster:
    cluster_id: int
    files: list[str]           # List of file paths
    centroid: np.ndarray       # Mean embedding of cluster
    coherence: float           # Internal similarity score
    label: str | None          # Optional human-readable label (from LLM or keywords)
```

#### Key Functions

| Function | Description |
|----------|-------------|
| `load_inbox_files(inbox_path: str, all_embeddings: dict) -> dict[str, np.ndarray]` | Extract embeddings for files within inbox directory |
| `cluster_files(embeddings: dict, distance_threshold: float = 0.3) -> list[FileCluster]` | Group similar files using agglomerative clustering |
| `compute_cluster_centroid(cluster: FileCluster, embeddings: dict) -> np.ndarray` | Compute mean embedding for cluster |
| `auto_label_cluster(cluster: FileCluster, file_contents: dict) -> str` | Generate label from common keywords or first few words |

#### Algorithm

```python
from sklearn.cluster import AgglomerativeClustering

def cluster_files(embeddings: dict[str, np.ndarray], distance_threshold: float = 0.3):
    paths = list(embeddings.keys())
    X = np.array([embeddings[p] for p in paths])
    
    # Normalize for cosine distance
    X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
    
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric='cosine',
        linkage='average',
        distance_threshold=distance_threshold
    )
    labels = clustering.fit_predict(X_normalized)
    
    # Group files by cluster
    clusters = {}
    for path, label in zip(paths, labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(path)
    
    # Create FileCluster objects
    result = []
    for cluster_id, files in clusters.items():
        centroid = np.mean([embeddings[f] for f in files], axis=0)
        centroid /= np.linalg.norm(centroid)  # Normalize
        
        result.append(FileCluster(
            cluster_id=cluster_id,
            files=files,
            centroid=centroid,
            coherence=compute_internal_coherence(files, embeddings),
            label=None
        ))
    
    return result
```

---

## Phase 2: Folder Matching

### Goal
For each cluster, find the most suitable destination folders **outside** the working directory.

### File: `suggestions.py` (Modified)

#### Key Changes from v1

1. **Exclude inbox from candidate folders**: When computing folder embeddings, skip folders under the inbox path
2. **Match cluster centroids** instead of individual files
3. **Batch suggestions**: One suggestion per cluster, not per file

#### New Functions

| Function | Description |
|----------|-------------|
| `build_destination_tree(embeddings: dict, exclude_prefix: str) -> FolderNode` | Build tree excluding inbox directory |
| `match_cluster_to_folders(cluster: FileCluster, folder_embs: dict, k: int = 5) -> list[tuple[str, float]]` | Find top-K folders similar to cluster centroid |
| `generate_cluster_suggestions(clusters: list[FileCluster], folder_embs: dict) -> list[ClusterSuggestion]` | Create batch move suggestions |

#### Data Structure

```python
@dataclass
class ClusterSuggestion:
    cluster_id: int
    files: list[str]              # Files in this cluster
    cluster_label: str | None     # Human-readable description
    candidates: list[tuple[str, float]]  # [(folder_path, similarity), ...]
```

---

## Phase 3: Main Orchestrator

### File: `inbox_organizer.py`

#### Workflow

```
1. Load all embeddings from doc_emb.json
2. Extract files matching inbox prefix (e.g., "FuckHere/")
3. Cluster inbox files by similarity
4. Build folder tree EXCLUDING inbox
5. Compute folder embeddings (or load from dir_emb.json)
6. For each cluster, find matching destination folders
7. Output suggestions (JSON + human-readable)
```

#### CLI Interface

```bash
# Analyze inbox and suggest destinations
python inbox_organizer.py --inbox "FuckHere" --threshold 0.3 --top-k 3

# Adjust clustering sensitivity
python inbox_organizer.py --inbox "Notes/FuckHere" --threshold 0.5  # Fewer, larger clusters
python inbox_organizer.py --inbox "Notes/FuckHere" --threshold 0.2  # More, smaller clusters

# Output options
python inbox_organizer.py --inbox "FuckHere" --output suggestions.json
python inbox_organizer.py --inbox "FuckHere" --output -  # stdout
```

---

## Output Format

### JSON

```json
{
  "analysis_date": "2025-12-11T00:30:00",
  "inbox_path": "Notes/FuckHere",
  "total_files": 45,
  "cluster_count": 8,
  "distance_threshold": 0.3,
  "clusters": [
    {
      "cluster_id": 0,
      "file_count": 6,
      "label": "Python async/await",
      "coherence": 0.85,
      "files": [
        "Notes/FuckHere/async-basics.md",
        "Notes/FuckHere/await-patterns.md",
        "Notes/FuckHere/asyncio-tutorial.md",
        ...
      ],
      "suggested_destinations": [
        {"folder": "Notes/Coding/Python/Async", "similarity": 0.87},
        {"folder": "Notes/Coding/Python/Advanced", "similarity": 0.76},
        {"folder": "Notes/Coding/Python", "similarity": 0.72}
      ]
    },
    {
      "cluster_id": 1,
      "file_count": 3,
      "label": "Docker containers",
      "coherence": 0.92,
      "files": [...],
      "suggested_destinations": [...]
    }
  ]
}
```

### Human-Readable Output

```
════════════════════════════════════════════════════════════════════
📥 Inbox Analysis: Notes/FuckHere
════════════════════════════════════════════════════════════════════

Found 45 files → Grouped into 8 clusters

────────────────────────────────────────────────────────────────────
Cluster #1: "Python async/await" (6 files, coherence: 0.85)
────────────────────────────────────────────────────────────────────
  📄 async-basics.md
  📄 await-patterns.md
  📄 asyncio-tutorial.md
  📄 ...

  ➡️  Suggested destinations:
      1. 📁 Notes/Coding/Python/Async (0.87)
      2. 📁 Notes/Coding/Python/Advanced (0.76)
      3. 📁 Notes/Coding/Python (0.72)

────────────────────────────────────────────────────────────────────
Cluster #2: "Docker containers" (3 files, coherence: 0.92)
────────────────────────────────────────────────────────────────────
  ...
```

---

## File Structure (Updated)

```
src/
├── data/
│   ├── data.json           # Indexed files
│   ├── doc_emb.json        # File embeddings
│   ├── dir_emb.json        # Folder embeddings
│   └── db.db               # SQLite database
├── folder_tree.py          # (Reuse) Tree structure
├── discrepancy.py          # (v1 - optional, can keep)
├── suggestions.py          # (Reuse with modifications)
├── clustering.py           # NEW: File clustering
├── inbox_organizer.py      # NEW: Main orchestrator
├── vault_organizer.py      # (v1 - can keep as alternative)
└── PLAN2.md                # This plan
```

---

## Implementation Steps

| Step | Task | File |
|------|------|------|
| 1 | Create clustering module with agglomerative clustering | `clustering.py` |
| 2 | Modify `folder_tree.py` to support path exclusion | `folder_tree.py` |
| 3 | Update `suggestions.py` for cluster-based matching | `suggestions.py` |
| 4 | Create main orchestrator with CLI | `inbox_organizer.py` |
| 5 | Test with actual inbox data | - |

---

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--inbox` | (required) | Path prefix for working directory (e.g., `"FuckHere"` or `"Notes/FuckHere"`) |
| `--threshold` | `0.3` | Clustering distance threshold (lower = more clusters) |
| `--min-cluster` | `1` | Minimum files per cluster to suggest |
| `--top-k` | `3` | Number of destination suggestions per cluster |
| `--min-similarity` | `0.5` | Minimum folder similarity to include |

---

## Example Use Case

**Before:**
```
FuckHere/
├── docker-compose-notes.md      ─┐
├── kubernetes-basics.md          ├─ Cluster A → Notes/IT/Containers/
├── container-networking.md      ─┘
├── pytorch-tensors.md           ─┐
├── neural-network-basics.md      ├─ Cluster B → Notes/AI/DL/
├── backprop-explained.md        ─┘
└── random-thoughts.md           ─── Cluster C (singleton) → ???
```

**After running:**
```bash
python inbox_organizer.py --inbox "FuckHere"

# Output:
Cluster 1 (3 files): docker-compose-notes.md, kubernetes-basics.md, container-networking.md
  → Suggested: Notes/IT/Containers/ (0.89 similarity)

Cluster 2 (3 files): pytorch-tensors.md, neural-network-basics.md, backprop-explained.md  
  → Suggested: Notes/AI/DL/ (0.91 similarity)

Cluster 3 (1 file): random-thoughts.md
  → Suggested: Notes/MyMe/Journal/ (0.52 similarity)
```

---

## Ready to Proceed?

This plan changes the approach from "detect outliers everywhere" to "organize a specific inbox folder". 

Key benefits:
- **Focused workflow**: Act on one directory at a time
- **Batch operations**: Move related files together
- **Clearer suggestions**: Cluster → destination mapping is more actionable

Let me know when you'd like me to implement this!
