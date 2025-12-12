# ChangeLog

## 2025-12-11

### Added new inbox organization workflow
- **`clustering.py`**: Implements Agglomerative Clustering (cosine distance) to group semantically similar files within a specified working directory. Includes utilities for computing internal coherence, auto‑labeling clusters, and pretty‑printing results.
- **`inbox_organizer.py`**: Main CLI orchestrator that:
  1. Loads all file embeddings from `data/doc_emb.json`.
  2. Extracts files belonging to a user‑specified inbox folder (e.g., `FuckHere`).
  3. Clusters those files using `clustering.py` (configurable distance threshold).
  4. Builds a folder‑tree (re‑using `folder_tree.py`) while excluding the inbox path.
  5. Computes or loads folder embeddings (`data/dir_emb.json`).
  6. Matches each cluster centroid to the most similar destination folders outside the inbox.
  7. Generates a detailed JSON report (`data/inbox_suggestions.json` and `data/inbox_suggestions_t05.json` for a coarser threshold) and a human‑readable console output.
- Added command‑line options for threshold tuning, quiet mode, output destination, and move‑command generation.

### Enhancements to existing modules
- **`clustering.py`** introduces:
  - `FileCluster` dataclass with centroid, coherence, and optional label.
  - Functions for cosine similarity, internal coherence, auto‑labeling, and statistics.
- Updated **`folder_tree.py`** (no functional change) is now referenced for building the hierarchical tree and computing folder embeddings.

### Generated artifacts
- `data/inbox_suggestions.json` – 207 clusters (threshold 0.35) with suggested destination folders.
- `data/inbox_suggestions_t05.json` – 52 larger clusters (threshold 0.5) for a higher‑level view.
- `data/dir_emb.json` – folder embeddings (unchanged, reused from previous workflow).

### Documentation
- Created `PLAN2.md` outlining the new inbox‑organizer design (later removed after implementation).

### Overall impact
- Provides a focused "inbox zero" tool that automatically groups and relocates scattered files in a designated working directory.
- Enables batch moves based on semantic similarity, dramatically reducing manual organization effort.
- Extensible CLI allows fine‑grained control over clustering granularity and output format.

---

*All changes were made on 2025‑12‑11.*
