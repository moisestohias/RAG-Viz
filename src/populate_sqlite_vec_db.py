"""
This script reads data stored in JSON format, calculates vector embeddings, and populates the data in SQLite database. It leverages the `sqlite-vec` extension to store the text embeddings for efficient vector similarity search, while also storing the raw comment details in a separate table.
"""

import sqlite3, json, struct
from contextlib import contextmanager
from embedder import EMBEDDING_MODELS, embed
from helper_utils import os, expand_full_path_and_ensure_file_exist, expand_full_path

EMBEDDING_DIM = 1024
MAX_TEXT_LEN = 200

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

def populate_database_with_embedding(data: list[dict], 
                                     db_name: str = "db.db", 
                                     limit_long_text:bool=False,  # ❗ BEAWRE
                                     MAX_TEXT_LEN:int=MAX_TEXT_LEN):
    if not data:
        return 0

    inserted_count = 0
    with init_sqlite_vec(db_name) as conn:
        conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_emb USING vec0(
                id TEXT PRIMARY KEY,
                document_embedding FLOAT[{EMBEDDING_DIM}]
            );
            """
        )

        # Get IDs already in the table that have a non-NULL embedding to avoid recomputing
        cur = conn.cursor()
        try:
            cur.execute("SELECT id FROM vec_emb WHERE document_embedding IS NOT NULL;")
            existing_ids = {row[0] for row in cur.fetchall()}
        finally:
            cur.close()

        # Insert only new rows
        to_insert = []
        for item in data:
            _id = item.get("id")
            if _id is None:
                raise ValueError("Each item must have an 'id' field.")
            if _id in existing_ids:
                continue  # skip existing rows

            text = (item.get("text") or "").strip()

            if not text: continue # skip empty 
            # Limit long text - we can infere topic by reading first dozone words.
            if limit_long_text: 
              text = text if len(text.split()) <= MAX_TEXT_LEN else " ".join(text.split()[:MAX_TEXT_LEN])

            embedding = embed(text)
            if not (isinstance(embedding, list) and len(embedding) == EMBEDDING_DIM):
                raise ValueError(f"Embedding for id={_id} must be a list of length {EMBEDDING_DIM}")

            to_insert.append((_id, serialize_f32(embedding)))

        if not to_insert:
            return 0

        # Insert new rows
        cur = conn.cursor()
        try:
            cur.executemany(
                "INSERT INTO vec_emb(id, document_embedding) VALUES(?, ?);",
                to_insert
            )
        finally:
            cur.close()

        inserted_count = len(to_insert)

    return inserted_count


def populate_database_text(data:list[dict], db_name:str, table_name:str):
  db_name = expand_full_path_and_ensure_file_exist(db_name)

  with sqlite3.connect(db_name) as conn:
    conn.execute("PRAGMA foreign_keys = ON;")

    # Create table
    conn.execute('''
    CREATE TABLE IF NOT EXISTS files (
      id TEXT PRIMARY KEY,
      text TEXT,
    )
    ''')

    cur = conn.cursor()
    try:
      # Insert data - Note; you can replace REPLACE with IGNORE 
      conn.executemany('''
      INSERT OR REPLACE INTO files (id, text)
      VALUES (?, ?)
      ''', [(item['id'], item['text']) for item in data])
    finally:
        cur.close()

def populate_database(data, db_name:str):
  populate_database_with_embedding(data, db_name)
  populate_database_text(data, db_name)


def main():
  with init_sqlite_vec("db.db") as conn:
    # sqlite-vec is already loaded
    cursor = conn.execute("SELECT * from vec_emb limit 2;")
    results = cursor.fetchall()
    for id, emb in results: print(id, deserialize_f32(emb))
  ## Load data
  # data_path = "data/data.json"
  # expand_full_path_and_ensure_file_exist(data_path)

  # with open(data_path) as f: data = json.load(f)
  # db_name:str = "db.db"
  # populate_database(data, db_name)


if __name__ == '__main__':
  main()