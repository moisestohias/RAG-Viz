"""
This script reads data stored in JSON format, calculates vector embeddings, and populates the data in SQLite database. It leverages the `sqlite-vec` extension to store the text embeddings for efficient vector similarity search, while also storing the raw comment details in a separate table.
"""

import sqlite3, json
from embedder import EMBEDDING_MODELS, embed
from helper_utils import os, expand_full_path_and_ensure_file_exist, expand_full_path

EMBEDDING_DIM = 1024
MAX_TEXT_LEN = 200

def populate_db_with_embedding(data: list[dict],
                                     db_name: str = "db.db",
                                     limit_long_text:bool=False,  # ❗ BEAWRE - Keep it false
                                     MAX_TEXT_LEN:int=MAX_TEXT_LEN,
                                     batch_size:int=10):
    if not data: return 0

    # Calculate total items to process (excluding existing IDs)
    with init_sqlite_vec(db_name) as conn:
        cur = conn.cursor()
        try:
            cur.execute("SELECT id FROM vec_emb WHERE document_embedding IS NOT NULL;")
            existing_ids = {row[0] for row in cur.fetchall()}
        finally: cur.close()

    # Count total items to process
    total_items = sum(1 for k, v in data.items() if k not in existing_ids and (v.get("content") or "").strip())

    inserted_count = 0
    processed_count = 0

    with init_sqlite_vec(db_name) as conn:
        conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_emb USING vec0(
                id TEXT PRIMARY KEY,
                document_embedding FLOAT[{EMBEDDING_DIM}]
            );
            """
        )

        # Process items in batches to write to DB every N files
        batch_to_insert = []
        for k, v in data.items():
            _id, text = k, v["content"]
            if _id in existing_ids:
                continue  # skip existing rows

            if not text: continue # skip empty
            # Limit long text - we can infere topic by reading first dozone words.
            if limit_long_text:
              text = text if len(text.split()) <= MAX_TEXT_LEN else " ".join(text.split()[:MAX_TEXT_LEN])

            embedding = embed(text)
            if not (isinstance(embedding, list) and len(embedding) == EMBEDDING_DIM):
                raise ValueError(f"Embedding for id={_id} must be a list of length {EMBEDDING_DIM}")

            batch_to_insert.append((_id, serialize_f32(embedding)))
            processed_count += 1

            # Write to DB every N=10 files
            if len(batch_to_insert) >= batch_size:
                cur = conn.cursor()
                try:
                    cur.executemany(
                        "INSERT INTO vec_emb(id, document_embedding) VALUES(?, ?);",
                        batch_to_insert
                    )
                    conn.commit()  # Explicitly commit the transaction
                except Exception as e:
                    print(f"Error inserting batch: {e}")
                    conn.rollback()  # Rollback on error
                    raise
                finally:
                    cur.close()

                inserted_count += len(batch_to_insert)

                # Print progress after each batch
                progress_percentage = (processed_count / total_items) * 100 if total_items > 0 else 0
                print(f"Progress: {processed_count}/{total_items} items processed ({progress_percentage:.2f}%) - Batch inserted: {len(batch_to_insert)} entries")

                batch_to_insert = []  # Reset batch

        # Insert any remaining items in the final batch
        if batch_to_insert:
            cur = conn.cursor()
            try:
                cur.executemany(
                    "INSERT INTO vec_emb(id, document_embedding) VALUES(?, ?);",
                    batch_to_insert
                )
                conn.commit()  # Explicitly commit the final transaction
            except Exception as e:
                print(f"Error inserting final batch: {e}")
                conn.rollback()  # Rollback on error
                raise
            finally:
                cur.close()

            inserted_count += len(batch_to_insert)

            # Print final progress
            progress_percentage = (processed_count / total_items) * 100 if total_items > 0 else 0
            print(f"Final: {processed_count}/{total_items} items processed ({progress_percentage:.2f}%) - Final batch inserted: {len(batch_to_insert)} entries")

    return inserted_count


def populate_db_with_precomputed_embeddings(data: dict, doc_emb: dict, db_name: str = "db.db"):
    if not data or not doc_emb:
        return 0

    with init_sqlite_vec(db_name) as conn:
        conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_emb USING vec0(
                id TEXT PRIMARY KEY,
                document_embedding FLOAT[{EMBEDDING_DIM}]
            );
            """
        )
        
        cur = conn.cursor()
        cur.execute("SELECT id FROM vec_emb;")
        existing_ids = {row[0] for row in cur.fetchall()}
        
        inserted_count = 0
        for _id in data:
            if _id in existing_ids or _id not in doc_emb: continue
            cur.execute(
                "INSERT INTO vec_emb(id, document_embedding) VALUES(?, ?);",
                (_id, serialize_f32(doc_emb[_id]))
            )
            inserted_count += 1
        
        conn.commit()
        cur.close()

    return inserted_count

def populate_db_text(data:dict, db_name:str):
  db_name = expand_full_path_and_ensure_file_exist(db_name)

  with sqlite3.connect(db_name) as conn:
    conn.execute("PRAGMA foreign_keys = ON;")

    # Create table
    conn.execute('''
    CREATE TABLE IF NOT EXISTS files (
      id TEXT PRIMARY KEY,
      text TEXT
    )
    ''')

    cur = conn.cursor()
    try:
      # Insert data - Note; you can replace REPLACE with IGNORE
      conn.executemany('''
      INSERT OR REPLACE INTO files (id, text)
      VALUES (?, ?)
      ''', [(k, v["content"]) for k, v in data.items()])
    finally:
        cur.close()

def populate_db(data, db_name:str, batch_size:int=10):
  populate_db_text(data, db_name)
  # populate_db_with_precomputed_embeddings(data, db_name, batch_size=batch_size)
  # populate_db_with_embedding(data, db_name, batch_size=batch_size)


def main():
  # # Load data
  # data_path = "data/data.json"
  # doc_emb = "data/doc_emb.json"
  # expand_full_path_and_ensure_file_exist(data_path)
  # expand_full_path_and_ensure_file_exist(doc_emb)

  # with open(data_path) as f: data = json.load(f)
  # with open(doc_emb) as f: doc_emb = json.load(f)

  # db_name:str = "data/db.db"
  # batch_size:int = 10
  # populate_db(data, db_name, batch_size)
  # populate_db_with_precomputed_embeddings(data, doc_emb, db_name)

  # with init_sqlite_vec("data/db.db") as conn:
  #   # sqlite-vec is already loaded
  #   cursor = conn.execute("SELECT * from vec_emb limit 2;")
  #   results = cursor.fetchall()
  #   for id, emb in results: print(id, deserialize_f32(emb))
  pass

if __name__ == '__main__':
  main()