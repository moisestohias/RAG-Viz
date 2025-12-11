# Guide: Storing Reduced-Dimensional Embeddings (UMAP Projections) in SQLite with sqlite-vec

## Overview
This guide outlines approaches for storing UMAP-projected embeddings (typically 2D or 3D) in your existing SQLite database alongside the original high-dimensional embeddings. This enables faster visualization and clustering of document embeddings without requiring repeated dimensionality reduction computations.

## Prerequisites
- Working knowledge of UMAP dimensionality reduction technique
- Understanding of the existing `populate_sqlite_vec_db.py` workflow
- Basic SQL knowledge, particularly regarding SQLite and the sqlite-vec extension

## Approaches to Store Reduced Embeddings

### Approach 1: Add Separate Columns to Existing Table

The most straightforward approach is to add columns for reduced-dimensional embeddings to the existing `vec_emb` table.

#### Step 1: Modify the Database Schema
Update the table schema to include columns for 2D and 3D UMAP projections:

```sql
-- Add new columns for reduced-dimension embeddings
ALTER TABLE vec_emb ADD COLUMN umap_2d FLOAT[2];
ALTER TABLE vec_emb ADD COLUMN umap_3d FLOAT[3];
```

#### Step 2: Update Your Code to Populate the New Columns
Modify `populate_sqlite_vec_db.py` to include reduced embeddings after computation:

```python
def populate_reduced_embeddings(db_name: str = "db.db", n_neighbors: int = 4, min_dist: float = 0.1, metric: str = "euclidean"):
    """Compute and store UMAP-reduced embeddings in the database"""
    from umap import UMAP
    import numpy as np
    from helper_utils import deserialize_f32, serialize_f32, load_embeddings_from_db
    
    # Load existing full-dimensional embeddings
    file_paths, emb_vectors = load_embeddings_from_db(db_name)
    
    # Compute UMAP projections
    reducer_2d = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
    reducer_3d = UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
    
    projections_2d = reducer_2d.fit_transform(emb_vectors)
    projections_3d = reducer_3d.fit_transform(emb_vectors)
    
    # Normalize to [-1, 1] range (optional)
    # projections_2d = (projections_2d - projections_2d.min(axis=0)) / (projections_2d.max(axis=0) - projections_2d.min(axis=0)) * 2 - 1
    # projections_3d = (projections_3d - projections_3d.min(axis=0)) / (projections_3d.max(axis=0) - projections_3d.min(axis=0)) * 2 - 1
    
    # Connect to database and update the table
    with init_sqlite_vec(db_name) as conn:
        cur = conn.cursor()
        
        # Prepare update statements
        for i, file_path in enumerate(file_paths):
            p2d_bytes = serialize_f32(projections_2d[i].tolist())
            p3d_bytes = serialize_f32(projections_3d[i].tolist())
            
            cur.execute(
                "UPDATE vec_emb SET umap_2d = ?, umap_3d = ? WHERE id = ?;",
                (p2d_bytes, p3d_bytes, file_path)
            )
        
        conn.commit()
        cur.close()
```

#### Step 3: Query Reduced Embeddings
To retrieve the reduced embeddings for visualization:

```python
def get_reduced_embeddings(db_name: str = "db.db", dims: int = 3):
    """Retrieve reduced-dimensional embeddings from the database"""
    from helper_utils import deserialize_f32
    
    with init_sqlite_vec(db_name) as conn:
        cur = conn.cursor()
        
        if dims == 2:
            cur.execute("SELECT id, umap_2d FROM vec_emb WHERE umap_2d IS NOT NULL;")
        elif dims == 3:
            cur.execute("SELECT id, umap_3d FROM vec_emb WHERE umap_3d IS NOT NULL;")
        else:
            raise ValueError("dims must be either 2 or 3")
        
        results = cur.fetchall()
        file_paths, reduced_embeddings = zip(*[(row[0], deserialize_f32(row[1])) for row in results])
        
        return list(file_paths), np.array(reduced_embeddings, dtype=np.float32)
```

### Approach 2: Create a Separate Table for Reduced Embeddings

For better organization and separation of concerns, you can maintain a separate table for reduced embeddings:

#### Step 1: Create the New Table
```sql
CREATE VIRTUAL TABLE vec_reduced USING vec0(
    id TEXT PRIMARY KEY,
    umap_2d FLOAT[2],
    umap_3d FLOAT[3]
);
-- Note: SQLite virtual tables, including those created with sqlite-vec, don't support
-- foreign key constraints directly in their definition. Foreign key relationships
-- must be enforced at the application level.
```

#### Step 2: Populate the New Table with UMAP Projections
```python
def populate_reduced_embeddings_table(db_name: str = "db.db", n_neighbors: int = 4, min_dist: float = 0.1, metric: str = "euclidean"):
    """Compute and store UMAP-reduced embeddings in a separate table"""
    from umap import UMAP
    import numpy as np
    from helper_utils import deserialize_f32, serialize_f32, load_embeddings_from_db
    
    # Load existing full-dimensional embeddings
    file_paths, emb_vectors = load_embeddings_from_db(db_name)
    
    # Compute UMAP projections
    reducer_2d = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
    reducer_3d = UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
    
    projections_2d = reducer_2d.fit_transform(emb_vectors)
    projections_3d = reducer_3d.fit_transform(emb_vectors)
    
    # Connect to database and create the new table
    with init_sqlite_vec(db_name) as conn:
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_reduced USING vec0(
                id TEXT PRIMARY KEY,
                umap_2d FLOAT[2],
                umap_3d FLOAT[3]
            );
            -- Note: SQLite virtual tables, including those created with sqlite-vec,
            -- don't support foreign key constraints directly in their definition.
            -- Foreign key relationships must be enforced at the application level.
            """
        )
        
        cur = conn.cursor()
        
        # Prepare data for insertion
        batch_to_insert = []
        for i, file_path in enumerate(file_paths):
            p2d_bytes = serialize_f32(projections_2d[i].tolist())
            p3d_bytes = serialize_f32(projections_3d[i].tolist())
            batch_to_insert.append((file_path, p2d_bytes, p3d_bytes))
        
        # Insert in batches for efficiency
        cur.executemany(
            "INSERT OR REPLACE INTO vec_reduced(id, umap_2d, umap_3d) VALUES(?, ?, ?);",
            batch_to_insert
        )
        
        conn.commit()
        cur.close()
```

#### Step 3: Query from the Separate Table
```python
def get_reduced_embeddings_from_separate_table(db_name: str = "db.db", dims: int = 3):
    """Retrieve reduced-dimensional embeddings from the separate table"""
    from helper_utils import deserialize_f32
    
    with init_sqlite_vec(db_name) as conn:
        cur = conn.cursor()
        
        if dims == 2:
            cur.execute("SELECT id, umap_2d FROM vec_reduced WHERE umap_2d IS NOT NULL;")
        elif dims == 3:
            cur.execute("SELECT id, umap_3d FROM vec_reduced WHERE umap_3d IS NOT NULL;")
        else:
            raise ValueError("dims must be either 2 or 3")
        
        results = cur.fetchall()
        file_paths, reduced_embeddings = zip(*[(row[0], deserialize_f32(row[1])) for row in results])
        
        return list(file_paths), np.array(reduced_embeddings, dtype=np.float32)
```

### Approach 3: Hybrid Approach with Metadata Storage

Store the UMAP parameters along with the projections to ensure reproducibility:

#### Step 1: Create Table with Parameters
```sql
CREATE VIRTUAL TABLE vec_reduced_meta USING vec0(
    id TEXT PRIMARY KEY,
    umap_2d FLOAT[2],
    umap_3d FLOAT[3],
    params_json TEXT,  -- JSON string storing UMAP parameters
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
-- Note: SQLite virtual tables, including those created with sqlite-vec, don't support
-- foreign key constraints directly in their definition. Foreign key relationships
-- must be enforced at the application level.
```

#### Step 2: Implementation with Parameter Tracking
```python
import json
from datetime import datetime

def populate_reduced_embeddings_with_metadata(db_name: str = "db.db", n_neighbors: int = 4, min_dist: float = 0.1, metric: str = "euclidean"):
    """Compute and store UMAP-reduced embeddings with metadata"""
    from umap import UMAP
    import numpy as np
    from helper_utils import deserialize_f32, serialize_f32, load_embeddings_from_db
    
    # Keep track of parameters used
    params = {
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "metric": metric,
        "computed_at": datetime.now().isoformat()
    }
    
    # Load existing full-dimensional embeddings
    file_paths, emb_vectors = load_embeddings_from_db(db_name)
    
    # Compute UMAP projections
    reducer_2d = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
    reducer_3d = UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
    
    projections_2d = reducer_2d.fit_transform(emb_vectors)
    projections_3d = reducer_3d.fit_transform(emb_vectors)
    
    # Connect to database and create the new table
    with init_sqlite_vec(db_name) as conn:
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_reduced_meta USING vec0(
                id TEXT PRIMARY KEY,
                umap_2d FLOAT[2],
                umap_3d FLOAT[3],
                params_json TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            -- Note: SQLite virtual tables, including those created with sqlite-vec,
            -- don't support foreign key constraints directly in their definition.
            -- Foreign key relationships must be enforced at the application level.
            """
        )
        
        cur = conn.cursor()
        
        # Prepare data for insertion
        batch_to_insert = []
        params_str = json.dumps(params)
        for i, file_path in enumerate(file_paths):
            p2d_bytes = serialize_f32(projections_2d[i].tolist())
            p3d_bytes = serialize_f32(projections_3d[i].tolist())
            batch_to_insert.append((file_path, p2d_bytes, p3d_bytes, params_str))
        
        # Insert in batches for efficiency
        cur.executemany(
            "INSERT OR REPLACE INTO vec_reduced_meta(id, umap_2d, umap_3d, params_json) VALUES(?, ?, ?, ?);",
            batch_to_insert
        )
        
        conn.commit()
        cur.close()
```

## Recommendations

### Recommended Approach: Approach 2 (Separate Table)
For most use cases, **Approach 2** (separate table) is recommended because:

1. **Clear Separation**: Keeps high-dimensional and low-dimensional embeddings organized separately
2. **Flexibility**: Allows for different UMAP parameters without bloating the main table
3. **Performance**: Reduces the size of the main embedding table for similarity searches
4. **Maintainability**: Easier to update UMAP parameters and recompute projections without affecting the main table

### When to Use Each Approach:
- **Approach 1**: Best for simple projects where you only need one set of reduced embeddings
- **Approach 2**: Ideal for most applications where you want clean separation and flexibility
- **Approach 3**: Useful when you need to track multiple UMAP runs with different parameters or need reproducibility

### Enforcing Relationships at the Application Level
Since SQLite virtual tables do not support foreign key constraints directly, it's important to maintain data consistency at the application level:

1. **Always verify referenced records exist** before inserting into the reduced embedding table
2. **Handle deletions properly** by removing corresponding entries in both tables
3. **Keep track of relationships** in your application logic to maintain referential integrity
4. **Consider using transactions** when modifying related records in multiple tables

## Integration with Existing Workflow

To integrate this with your existing workflow:

1. **After populating the database with high-dimensional embeddings**, run the UMAP computation:
   ```python
   # After running populate_db_with_embedding() or similar
   populate_reduced_embeddings_table(db_name="data/db.db", n_neighbors=4, min_dist=0.1, metric="euclidean")
   ```

2. **For visualization**, retrieve the reduced embeddings:
   ```python
   # For 3D visualization
   ids, projections_3d = get_reduced_embeddings_from_separate_table(dims=3)
   
   # Use with your favorite plotting library
   import matplotlib.pyplot as plt
   from mpl_toolkits.mplot3d import Axes3D
   
   fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')
   ax.scatter(projections_3d[:, 0], projections_3d[:, 1], projections_3d[:, 2])
   plt.show()
   ```

## Potential Issues and Solutions

### Issue 1: Maintaining Referential Integrity Without Foreign Keys
**Solution**: Since SQLite virtual tables don't support foreign key constraints, you must enforce relationships at the application level:

```python
def validate_and_maintain_relationships(db_name: str = "db.db"):
    """Validate and maintain referential integrity between tables"""

    with init_sqlite_vec(db_name) as conn:
        cursor = conn.cursor()

        # Check for orphaned records in vec_reduced that don't exist in vec_emb
        cursor.execute("""
            SELECT vr.id
            FROM vec_reduced vr
            LEFT JOIN vec_emb ve ON vr.id = ve.id
            WHERE ve.id IS NULL
        """)

        orphaned_ids = [row[0] for row in cursor.fetchall()]

        if orphaned_ids:
            print(f"Found {len(orphaned_ids)} orphaned records in vec_reduced. Cleaning up...")

            # Remove orphaned records
            placeholders = ','.join(['?' for _ in orphaned_ids])
            cursor.execute(f"DELETE FROM vec_reduced WHERE id IN ({placeholders})", orphaned_ids)
            conn.commit()

### Issue 2: Memory Constraints with Large Datasets
**Solution**: Process embeddings in chunks if you have memory limitations:
```python
def chunked_umap_processing(db_name, chunk_size=1000):
    """Process UMAP in chunks to handle large datasets"""
    import numpy as np
    from helper_utils import deserialize_f32, serialize_f32
    
    # First, get all IDs
    with init_sqlite_vec(db_name) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM vec_emb WHERE document_embedding IS NOT NULL;")
        all_ids = [row[0] for row in cursor.fetchall()]
    
    # Process in chunks
    for i in range(0, len(all_ids), chunk_size):
        chunk_ids = all_ids[i:i+chunk_size]
        
        # Get embeddings for this chunk
        with init_sqlite_vec(db_name) as conn:
            cursor = conn.cursor()
            placeholders = ','.join(['?' for _ in chunk_ids])
            cursor.execute(f"""
                SELECT id, document_embedding 
                FROM vec_emb 
                WHERE id IN ({placeholders})
            """, chunk_ids)
            
            results = cursor.fetchall()
            chunk_paths, chunk_embeddings = zip(*[(row[0], deserialize_f32(row[1])) for row in results])
            chunk_embeddings = np.array(chunk_embeddings, dtype=np.float32)
        
        # Compute UMAP for this chunk
        reducer = UMAP(n_components=3, n_neighbors=4, min_dist=0.1, metric="euclidean")
        chunk_projections = reducer.fit_transform(chunk_embeddings)
        
        # Update database with results
        with init_sqlite_vec(db_name) as conn:
            cur = conn.cursor()
            for j, path in enumerate(chunk_paths):
                proj_bytes = serialize_f32(chunk_projections[j].tolist())
                cur.execute("UPDATE vec_reduced SET umap_3d = ? WHERE id = ?", (proj_bytes, path))
            conn.commit()
            cur.close()
```

### Issue 3: Recomputing Projections After New Embeddings
**Solution**: Create a utility function to handle updates:
```python
def update_reduced_embeddings_for_new_entries(db_name: str = "db.db", **umap_kwargs):
    """Update reduced embeddings only for entries that don't have them yet"""
    from umap import UMAP
    import numpy as np
    from helper_utils import deserialize_f32, serialize_f32
    
    # Find entries that have embeddings but lack reduced embeddings
    with init_sqlite_vec(db_name) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT ve.id, ve.document_embedding 
            FROM vec_emb ve 
            LEFT JOIN vec_reduced vr ON ve.id = vr.id
            WHERE vr.id IS NULL AND ve.document_embedding IS NOT NULL
        """)
        results = cursor.fetchall()
        
        if not results:
            print("All entries have reduced embeddings already")
            return
        
        new_paths, new_embeddings = zip(*[(row[0], deserialize_f32(row[1])) for row in results])
        new_embeddings = np.array(new_embeddings, dtype=np.float32)
    
    # Get ALL embeddings to ensure consistent UMAP space (important!)
    all_paths, all_embeddings = load_embeddings_from_db(db_name)
    
    # Compute UMAP on ALL data to maintain consistency
    reducer = UMAP(n_components=3, **umap_kwargs)
    all_projections = reducer.fit_transform(all_embeddings)
    
    # Update database with new projections
    with init_sqlite_vec(db_name) as conn:
        cur = conn.cursor()
        for i, path in enumerate(all_paths):
            if path in new_paths:  # Only update newly added entries
                proj_bytes = serialize_f32(all_projections[i].tolist())
                cur.execute(
                    "INSERT OR REPLACE INTO vec_reduced(id, umap_3d) VALUES(?, ?);",
                    (path, proj_bytes)
                )
        conn.commit()
        cur.close()
```

This guide provides comprehensive approaches for storing reduced-dimensional embeddings in your SQLite database with sqlite-vec, enabling efficient visualization and analysis of your document embeddings.
