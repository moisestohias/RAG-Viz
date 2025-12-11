"""
Module for storing and retrieving reduced-dimensional embeddings (UMAP projections) in a separate SQLite table.
This follows Approach 2 from the Saving-Reduced-Embedding.md guide.
Note: The vec_reduced table has the same ID as vec_emb table to maintain logical relationship,
but the foreign key constraint is not enforced in the virtual table schema.
"""

import numpy as np
from umap import UMAP

from helper_utils import (
    deserialize_f32,
    serialize_f32,
    init_sqlite_vec,
    load_embeddings_from_db,
    expand_full_path
)


def populate_reduced_embeddings_table(
    db_name: str = "db.db", 
    n_neighbors: int = 4, 
    min_dist: float = 0.1, 
    metric: str = "euclidean"
) -> None:
    """Compute and store UMAP-reduced embeddings in a separate table"""
    
    # Load existing full-dimensional embeddings
    file_paths, emb_vectors = load_embeddings_from_db(db_name)
    
    if len(file_paths) == 0:
        print("No embeddings found in database to reduce.")
        return
    
    print(f"Computing UMAP projections for {len(file_paths)} embeddings...")
    
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
        
    print(f"Successfully stored reduced embeddings for {len(file_paths)} documents in vec_reduced table.")


def get_reduced_embeddings(
    db_name: str = "db.db", 
    dims: int = 3
) -> tuple[list, np.ndarray]:
    """Retrieve reduced-dimensional embeddings from the separate table"""
    
    if dims not in [2, 3]:
        raise ValueError("dims must be either 2 or 3")
    
    with init_sqlite_vec(db_name) as conn:
        cur = conn.cursor()
        
        if dims == 2:
            cur.execute("SELECT id, umap_2d FROM vec_reduced WHERE umap_2d IS NOT NULL;")
        elif dims == 3:
            cur.execute("SELECT id, umap_3d FROM vec_reduced WHERE umap_3d IS NOT NULL;")
        
        results = cur.fetchall()
        
        if not results:
            return [], np.array([])
        
        file_paths, reduced_embeddings = zip(*[(row[0], deserialize_f32(row[1])) for row in results])
        
        return list(file_paths), np.array(reduced_embeddings, dtype=np.float32)


def update_reduced_embeddings_for_new_entries(
    db_name: str = "db.db", 
    **umap_kwargs
) -> None:
    """Update reduced embeddings only for entries that don't have them yet"""
    
    from helper_utils import load_embeddings_from_db
    
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
    
    print(f"Recomputing UMAP for {len(all_embeddings)} total embeddings to update {len(new_paths)} new entries")
    
    # Compute UMAP on ALL data to maintain consistency
    reducer = UMAP(**umap_kwargs)
    all_projections = reducer.fit_transform(all_embeddings)
    
    # Update database with new projections
    with init_sqlite_vec(db_name) as conn:
        cur = conn.cursor()
        for i, path in enumerate(all_paths):
            if path in new_paths:  # Only update newly added entries
                # Get the corresponding 2D and 3D projections
                proj_2d = all_projections[i][:2]  # First 2 dimensions
                proj_3d = all_projections[i]      # All 3 dimensions
                
                proj_2d_bytes = serialize_f32(proj_2d.tolist())
                proj_3d_bytes = serialize_f32(proj_3d.tolist())
                
                cur.execute(
                    "INSERT OR REPLACE INTO vec_reduced(id, umap_2d, umap_3d) VALUES(?, ?, ?);",
                    (path, proj_2d_bytes, proj_3d_bytes)
                )
        conn.commit()
        cur.close()

    print(f"Updated reduced embeddings for {len(new_paths)} new entries")


def get_available_reduced_embeddings_count(db_name: str = "db.db") -> dict:
    """Get count of available reduced embeddings in the database"""
    
    with init_sqlite_vec(db_name) as conn:
        cur = conn.cursor()
        
        # Count 2D embeddings
        cur.execute("SELECT COUNT(*) FROM vec_reduced WHERE umap_2d IS NOT NULL;")
        count_2d = cur.fetchone()[0]
        
        # Count 3D embeddings
        cur.execute("SELECT COUNT(*) FROM vec_reduced WHERE umap_3d IS NOT NULL;")
        count_3d = cur.fetchone()[0]
        
        return {"2d": count_2d, "3d": count_3d}


if __name__ == "__main__":
    # Example usage
    db_path = "data/db.db"
    
    # Compute and store reduced embeddings
    populate_reduced_embeddings_table(db_path, n_neighbors=4, min_dist=0.1, metric="euclidean")
    
    # Retrieve 3D reduced embeddings for visualization
    ids, projections_3d = get_reduced_embeddings(db_path, dims=3)
    print(f"Retrieved {len(ids)} 3D reduced embeddings")
    
    # Check how many embeddings are available
    counts = get_available_reduced_embeddings_count(db_path)
    print(f"Available embeddings - 2D: {counts['2d']}, 3D: {counts['3d']}")