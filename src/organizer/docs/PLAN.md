# Vault Organizer - Implementation Plan

## Overview

This plan outlines the implementation of a **semantic vault organization system** that identifies thematically inconsistent folders and suggests file relocations based on embedding similarity.

---

## Current State

| Asset | Description |
|-------|-------------|
| `data/data.json` | 2923 indexed files with paths and content snippets |
| `data/doc_emb.json` | Pre-computed 1024-dim embeddings for each file |
| `data/db.db` | SQLite database with `files` (text) and `vec_emb` (embeddings) tables |
| `populate_sqlite_vec_db.py` | DB utilities, serialization, sqlite-vec integration |
| `embedder.py` | Ollama-based embedding generation via local API |
| `indexer.py` | Markdown file indexing utility |

---

## Implementation Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          vault_organizer.py                         │
│                         (Main Orchestrator)                         │
└─────────────────────────────────────────────────────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          ▼                        ▼                        ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  folder_tree.py  │    │  discrepancy.py  │    │  suggestions.py  │
│                  │    │                  │    │                  │
│ • Build tree     │    │ • Compute folder │    │ • Find similar   │
│ • Bottom-up      │    │   coherence      │    │   folders        │
│   aggregation    │    │ • Identify       │    │ • Generate move  │
│ • Folder         │    │   outlier files  │    │   proposals      │
│   embeddings     │    │                  │    │                  │
└──────────────────┘    └──────────────────┘    └──────────────────┘
          │                        │                        │
          └────────────────────────┼────────────────────────┘
                                   ▼
                    ┌──────────────────────────┐
                    │   populate_sqlite_vec_db │
                    │    (Existing DB Utils)   │
                    └──────────────────────────┘
```

---

## Phase 1: Folder Tree Construction & Embedding Aggregation

### Goal
Build an in-memory hierarchical representation of the vault and compute folder-level embeddings via **bottom-up aggregation**.

### File: `folder_tree.py`

#### Data Structures

```python
@dataclass
class FileNode:
    path: str                    # Relative path (e.g., "Notes/Python/async.md")
    embedding: np.ndarray        # 1024-dim vector
    parent: "FolderNode"

@dataclass  
class FolderNode:
    path: str                    # Folder path (e.g., "Notes/Python")
    files: list[FileNode]        # Direct child files
    subfolders: list["FolderNode"]  # Direct child folders
    embedding: np.ndarray | None # Aggregated embedding (computed bottom-up)
    parent: "FolderNode | None"
```

#### Key Functions

| Function | Description |
|----------|-------------|
| `build_tree(file_embeddings: dict) -> FolderNode` | Parse file paths into hierarchical tree structure |
| `compute_folder_embeddings(root: FolderNode) -> None` | **Recursive bottom-up** traversal: compute folder embedding as mean of all descendant embeddings |
| `get_all_folders(root: FolderNode) -> list[FolderNode]` | Flatten tree to list for iteration |

#### Algorithm: Bottom-Up Embedding Aggregation

```
def compute_folder_embeddings(folder: FolderNode):
    # 1. Recursively compute child folder embeddings first
    for subfolder in folder.subfolders:
        compute_folder_embeddings(subfolder)
    
    # 2. Collect all descendant embeddings
    all_embeddings = []
    for file in folder.files:
        all_embeddings.append(file.embedding)
    for subfolder in folder.subfolders:
        all_embeddings.append(subfolder.embedding)
    
    # 3. Compute folder embedding as mean
    if all_embeddings:
        folder.embedding = np.mean(all_embeddings, axis=0)
        # Normalize to unit vector for cosine similarity
        folder.embedding /= np.linalg.norm(folder.embedding)
```

---

## Phase 2: Semantic Discrepancy Detection

### Goal
Identify folders containing **thematically inconsistent** files by measuring how much individual file embeddings diverge from the folder's centroid.

### File: `discrepancy.py`

#### Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **File Deviation** | `1 - cosine_similarity(file_emb, folder_emb)` | How far a file is from its folder's semantic center |
| **Folder Coherence** | `mean(cosine_sim(file_emb, folder_emb)) for all files` | Overall thematic consistency (higher = better) |
| **Folder Variance** | `std(cosine_sim(file_emb, folder_emb))` | Spread of similarities (lower = more consistent) |

#### Key Functions

| Function | Description |
|----------|-------------|
| `compute_file_deviations(folder: FolderNode) -> dict[str, float]` | For each direct child file, compute deviation from folder centroid |
| `compute_folder_coherence(folder: FolderNode) -> float` | Mean cosine similarity of all direct files to folder centroid |
| `rank_incoherent_folders(folders: list[FolderNode]) -> list[tuple[FolderNode, float]]` | Sort folders by incoherence (ascending coherence) |
| `identify_outlier_files(folder: FolderNode, threshold: float = 0.3) -> list[FileNode]` | Files with deviation > threshold |

#### Outlier Detection Strategy

Using **z-score** or **IQR** (Interquartile Range) within each folder:

```python
def identify_outlier_files(folder: FolderNode, z_threshold: float = 2.0):
    deviations = [1 - cosine_sim(f.embedding, folder.embedding) for f in folder.files]
    mean_dev, std_dev = np.mean(deviations), np.std(deviations)
    
    outliers = []
    for file, dev in zip(folder.files, deviations):
        z_score = (dev - mean_dev) / std_dev if std_dev > 0 else 0
        if z_score > z_threshold:
            outliers.append((file, dev, z_score))
    
    return outliers
```

---

## Phase 3: Relocation Suggestions

### Goal
For outlier files, find semantically similar folders as potential relocation targets using **sqlite-vec** vector search.

### File: `suggestions.py`

#### Approach

1. **Store folder embeddings** in `data/dir_emb.json` (consistent with `doc_emb.json` pattern)
2. **Load folder embeddings** and compute cosine similarity to outlier files
3. **Filter** results to exclude the file's current folder and its ancestors

#### Key Functions

| Function | Description |
|----------|-------------|
| `save_folder_embeddings(folders: list[FolderNode], path: str = "data/dir_emb.json")` | Serialize folder embeddings to JSON |
| `load_folder_embeddings(path: str = "data/dir_emb.json") -> dict[str, list[float]]` | Load cached folder embeddings |
| `find_similar_folders(file: FileNode, folder_embs: dict, k: int = 5) -> list[tuple[str, float]]` | Compute top-K similar folders via cosine similarity |
| `generate_suggestions(outliers: list[FileNode], k: int = 3) -> list[Suggestion]` | Create actionable move proposals |

#### Storage Format (`dir_emb.json`)

```json
{
  "Notes/Python": [0.023, -0.145, ...],  // 1024-dim vector
  "Notes/Python/Advanced": [0.018, -0.132, ...],
  ...
}
```

#### Suggestion Data Structure

```python
@dataclass
class Suggestion:
    file_path: str                # File to move
    current_folder: str           # Current location
    deviation_score: float        # How "out of place" the file is
    candidates: list[tuple[str, float]]  # [(folder_path, similarity_score), ...]
```

---

## Phase 4: Main Orchestrator & CLI

### File: `vault_organizer.py`

#### Workflow

```
1. Load embeddings from database
2. Build folder tree structure
3. Compute folder embeddings (bottom-up)
4. Store folder embeddings to DB
5. Analyze each folder for coherence
6. Identify outlier files
7. Generate relocation suggestions
8. Output results (JSON / CLI report)
```

#### CLI Interface

```bash
# Analyze vault and output suggestions
python vault_organizer.py analyze --db data/db.db --threshold 2.0 --top-k 5

# Output formats
python vault_organizer.py analyze --output suggestions.json  # JSON
python vault_organizer.py analyze --output -                 # Stdout (human-readable)

# Preview mode (dry-run)
python vault_organizer.py preview --db data/db.db

# Execute moves (with confirmation)
python vault_organizer.py execute --suggestions suggestions.json --vault-root /path/to/vault
```

#### Output Format (JSON)

```json
{
  "analysis_date": "2025-12-10T22:57:00",
  "total_files": 2923,
  "total_folders": 412,
  "incoherent_folders": [
    {
      "path": "FuckeHere/random",
      "coherence_score": 0.42,
      "file_count": 15,
      "outliers": [
        {
          "file": "FuckeHere/random/python_decorators.md",
          "deviation": 0.58,
          "z_score": 2.3,
          "suggestions": [
            {"folder": "Notes/Python/Advanced", "similarity": 0.87},
            {"folder": "Notes/Programming/Patterns", "similarity": 0.72}
          ]
        }
      ]
    }
  ]
}
```

---

## File Structure (Final)

```
src/
├── data/
│   ├── data.json           # Indexed files
│   ├── doc_emb.json        # Pre-computed embeddings  
│   └── db.db               # SQLite database
├── embedder.py             # (Existing) Embedding generation
├── indexer.py              # (Existing) File indexer
├── helper_utils.py         # (Existing) Path utilities
├── populate_sqlite_vec_db.py  # (Existing) DB utilities
├── folder_tree.py          # NEW: Tree structure & aggregation
├── discrepancy.py          # NEW: Coherence analysis
├── suggestions.py          # NEW: Relocation proposals
├── vault_organizer.py      # NEW: Main orchestrator & CLI
├── Task.md                 # Task description
├── AGENTS.md               # Project context
├── clustering-guide.md     # Reference guide
└── PLAN.md                 # This plan
```

---

## Implementation Order

| Step | File | Tasks | Dependencies |
|------|------|-------|--------------|
| 1 | `folder_tree.py` | Tree construction, bottom-up aggregation | `numpy` |
| 2 | `discrepancy.py` | Coherence metrics, outlier detection | `folder_tree.py`, `numpy`, `scipy` |
| 3 | `suggestions.py` | Folder embedding storage, similarity search | `populate_sqlite_vec_db.py`, `folder_tree.py` |
| 4 | `vault_organizer.py` | CLI, orchestration, output formatting | All modules |

---

## Key Design Decisions

### 1. Why Use Mean for Folder Embeddings?

**Alternatives considered:**
- **Weighted mean** (by file size/importance): More complex, requires additional metadata
- **Centroid of clusters within folder**: Overkill for small folders
- **Median**: Less sensitive to outliers but computationally expensive for high-dim vectors

**Decision:** Simple mean is interpretable, fast, and effective. Normalizing to unit vectors ensures consistent cosine similarity comparisons.

### 2. Why Z-Score for Outlier Detection?

**Alternatives considered:**
- **Fixed threshold** (e.g., deviation > 0.5): Doesn't adapt to folder-specific distributions
- **IQR-based**: More robust but complex for small sample sizes
- **Isolation Forest / LOF**: Overkill for univariate deviation scores

**Decision:** Z-score is simple, interpretable, and adapts to each folder's internal distribution.

### 3. Why Store Folder Embeddings in DB?

**Alternatives considered:**
- **Compute on-the-fly**: Slower for repeated queries
- **In-memory only**: Fine for single run, but loses cached work

**Decision:** Storing in `vec_folders` table enables fast similarity searches via sqlite-vec's optimized index and persists computed work.

---

## Dependencies

```bash
# Core
pip install numpy scipy

# Already available
sqlite3  # Built-in
requests # For embedder.py

# Optional (for future enhancements)
pip install argparse  # Built-in, for CLI
pip install rich      # Pretty CLI output (optional)
```

---

## Testing Strategy

1. **Unit tests** for each module:
   - `folder_tree.py`: Tree construction from mock paths
   - `discrepancy.py`: Coherence calculation with synthetic embeddings
   - `suggestions.py`: Mock similarity search results

2. **Integration test**: 
   - Small subset of vault (50 files)
   - Verify end-to-end pipeline produces valid JSON output

3. **Validation**:
   - Manually review top-5 suggested moves
   - Confirm semantic relevance of suggestions

---

## Future Enhancements (Post-MVP)

- [ ] **Interactive mode**: TUI for reviewing/approving suggestions one-by-one
- [ ] **Undo system**: Track moves for rollback
- [ ] **Incremental updates**: Only recalculate affected folder embeddings after moves
- [ ] **Clustering integration**: Use UMAP+HDBSCAN to discover emergent categories
- [ ] **Folder creation**: Suggest creating new folders for uncategorized clusters

---

## Ready to Proceed?

Once you approve this plan, I will implement the modules in the following order:

1. **`folder_tree.py`** - Core tree structure and aggregation
2. **`discrepancy.py`** - Coherence analysis and outlier detection  
3. **`suggestions.py`** - Relocation proposal generation
4. **`vault_organizer.py`** - Main CLI and orchestration

Let me know if you'd like to adjust the approach, thresholds, or output format!
