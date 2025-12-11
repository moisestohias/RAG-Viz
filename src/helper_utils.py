import os
import sqlite3
import struct
import json
from contextlib import contextmanager


def expand_full_path(path:str):
  return os.path.expanduser(path) if path.startswith("~") else os.path.abspath(path)

def expand_full_path_and_ensure_file_exist(file_path:str):
  file_path = expand_full_path(file_path)
  if not os.path.exists(file_path):
    raise FileNotFoundError(f"File: {file_path!r} doesn't exist")
  return file_path


def serialize_f32(vector: list[float]) -> bytes:
    """serializes a list of floats into a compact "raw bytes" format"""
    return struct.pack("%sf" % len(vector), *vector)
def deserialize_f32(blob: bytes) -> list[float]:
    """Convert raw bytes back into a list of floats."""
    length = len(blob) // 4
    return list(struct.unpack(f"{length}f", blob))


def load_sqlite_vec_extension(conn:sqlite3.Connection, sqlite_vec_extension_path:str="~/.local/vec0.so"):
    sqlite_vec_extension_path = expand_full_path_and_ensure_file_exist(sqlite_vec_extension_path)
    conn.enable_load_extension(True)
    try: conn.load_extension(sqlite_vec_extension_path)
    finally: conn.enable_load_extension(False)
    return conn # not needed, modification happens in place - just nice for method chaining.

@contextmanager
def init_sqlite_vec(db_path: str = ":memory:") -> sqlite3.Connection:
    db_path = db_path = expand_full_path(db_path)
    if not os.path.exists(db_path): print("❗ No exisint database found, creating one")
    """Initialize an SQLite connection with sqlite-vec loaded."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    load_sqlite_vec_extension(conn)
    try: yield conn
    finally: conn.close()


def load_embeddings_from_db(db_path="data/db.db"):
    import numpy as np
    """Load file paths and embeddings from vec_emb table"""
    print("Loading embeddings from database...")
    with init_sqlite_vec(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, document_embedding 
            FROM vec_emb 
            WHERE document_embedding IS NOT NULL
        ''')
        results = cursor.fetchall()
    print(f"Found {len(results)} embeddings in database")
    file_paths, embeddings = zip(*[(row[0], deserialize_f32(row[1])) for row in results])
    return list(file_paths), np.array(embeddings, dtype=np.float32)


def load_data_from_db(db_path="data/db.db", use_content_snippets=True):
    import numpy as np

    """Load data and embeddings from SQLite database using existing functionality
    Args:
        db_path: Path to the SQLite database file
        use_content_snippets: If True, returns content snippets; if False, returns document titles (file paths)
    """
    print("Loading data from database...")

    with init_sqlite_vec(db_path) as conn:
        cursor = conn.cursor()

        # Query both the files and vec_emb tables to get matching records
        cursor.execute('''
            SELECT f.id, f.text, ve.document_embedding
            FROM files f
            INNER JOIN vec_emb ve ON f.id = ve.id
            WHERE ve.document_embedding IS NOT NULL
        ''')

        results = cursor.fetchall()
        print(f"Found {len(results)} matching entries in database")

        # Extract embeddings and content
        embedding_vectors = []
        texts_or_titles = []
        file_paths = []

        for row in results:
            file_id, text, emb_blob = row
            embedding = deserialize_f32(emb_blob)
            embedding_vectors.append(embedding)

            # Depending on the flag, return either content or title
            if use_content_snippets:
                texts_or_titles.append(text)
            else:
                texts_or_titles.append(file_id)  # Use file path as title

            file_paths.append(file_id)

    return np.array(embedding_vectors, dtype=np.float32), texts_or_titles, file_paths

def main():
  data = load_embeddings_from_db()
if __name__ == '__main__':
  main()
