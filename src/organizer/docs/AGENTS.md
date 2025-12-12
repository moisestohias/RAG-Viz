# Comprehensive Codebase Context Document

## 1. Project Purpose & Overview

This project implements a pipeline to create a semantic search-capable database. It processes text data from a JSON source, generates vector embeddings via a local API, and stores both the original text and the embeddings in a SQLite database.

The core functionality is provided by `populate_sqlite_vec_db.py`, which uses the `sqlite-vec` SQLite extension to enable efficient vector similarity searches. The system is designed to be run as a script to populate the database.

A separate utility, `indexer.py`, is also included. This script scans a directory for Markdown (`.md`) files and creates a JSON file containing their summaries, but it is not part of the main database population pipeline.

## 2. High-Level Architecture & Structure

The main application (`populate_sqlite_vec_db.py`) follows a clear, script-based workflow:

1.  **Database Initialization**: It connects to a SQLite database file (e.g., `data/db.db`) and loads the `sqlite-vec` extension from a specified binary file path (e.g., `~/.local/vec0.so`).
2.  **Data Ingestion**: The script is designed to read data from a JSON file.
3.  **Text Embedding**: For each piece of text, it calls the `embed` function from the `embedder.py` module. This function makes an HTTP request to a locally running service (like Ollama) to get a vector embedding.
4.  **Data Serialization**: The returned float vectors are serialized into a compact byte format for efficient storage in the database.
5.  **Database Population**: The script populates two tables:
    *   `files`: A standard table to store the ID and original text content.
    *   `vec_emb`: A virtual table powered by `sqlite-vec` to store the ID and the serialized vector embedding.

This architecture decouples the embedding model from the application itself, requiring a separate, running service to handle the embedding generation.

## 3. Major Components & Their Locations

| Component                    | Purpose                                                                   | Location(s)                       | Dependencies / Requirements                     |
| ---------------------------- | ------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------- |
| **DB Population Script**     | Orchestrates the entire embedding and database insertion process.         | `populate_sqlite_vec_db.py`       | `embedder.py`, `helper_utils.py`                |
| **Embedding Client**         | Sends text to a local API endpoint to get vector embeddings.              | `embedder.py`                     | `requests` (Python library)                     |
| **Path/File Utilities**      | Helper functions for path expansion and file existence checks.            | `helper_utils.py`                 | (Standard libraries)                            |
| **Vector Database Engine**   | The core vector search functionality within SQLite.                       | `~/.local/vec0.so` (example path) | A pre-compiled `sqlite-vec` binary file.        |
| **Markdown Indexer** (Utility) | Scans `.md` files and creates a JSON summary. Separate from the DB pipeline. | `indexer.py`                 | `PyYAML` (Python library, for frontmatter)      |

## 4. File-by-File Breakdown

### `populate_sqlite_vec_db.py`
*   **Primary Purpose**: The main script that drives the database population process.
*   **Key Components**:
    *   `init_sqlite_vec()`: A context manager that handles connecting to SQLite and loading the `sqlite-vec` binary extension.
    *   `populate_db_with_embedding()`: Reads data, calls the `embed()` function, and inserts the ID and serialized embedding into the `vec_emb` virtual table.
    *   `populate_db_text()`: Inserts the ID and raw text into the `files` table.
    *   `serialize_f32()` / `deserialize_f32()`: Utility functions to convert float vectors to and from bytes.
*   **Role in System**: This is the central orchestrator for creating the searchable database.

### `embedder.py`
*   **Primary Purpose**: Acts as a client to an external, local embedding service.
*   **Key Components**:
    *   `embed()`: Takes a string of text, formats it into a JSON payload, and POSTs it to a hardcoded API endpoint (`http://localhost:11434/api/embed`) to get embeddings.
    *   `@retry`, `@timing`: Decorators to add resilience and performance logging to the API calls.
*   **Role in System**: Decouples the application from the ML model. It requires a separate service (like Ollama) to be running to function.

### `helper_utils.py`
*   **Primary Purpose**: Provides file-system-related helper functions.
*   **Key Components**:
    *   `expand_full_path()`: Expands paths containing `~`.
    *   `expand_full_path_and_ensure_file_exist()`: Expands a path and verifies that the file exists, raising an error if it doesn't.
*   **Role in System**: Provides robust path handling for locating the database and `sqlite-vec` binary.

### `indexer.py`
*   **Primary Purpose**: A separate command-line utility to create an index of Markdown files. **This is not part of the main database pipeline.**
*   **Key Components**:
    *   `construct_md_json()`: Walks a directory, finds all `.md` files, extracts a snippet of their content (or YAML frontmatter `description`), and saves the results to `indexed.json`.
*   **Role in System**: A supplementary tool, likely for creating a searchable text index outside of the primary SQLite database.

## 5. Additional Context

*   **Python Dependencies**:
    *   `requests`: For making HTTP calls in `embedder.py`.
    *   `PyYAML`: Likely required for `indexer.py` to parse Markdown frontmatter.
*   **Runtime Dependencies**:
    *   **`sqlite-vec` Binary**: A compiled binary file of the `sqlite-vec` extension (e.g., `vec0.so`) must be available on the filesystem at a known location. This is **not** a Python package.
    *   **Local Embedding Service**: An API-compatible service (like Ollama) must be running and accessible at the endpoint specified in `embedder.py`.
*   **Execution Model**: The main functionality is script-based. One would run `python populate_sqlite_vec_db.py` to populate the database from a JSON data source. The script's `main` function currently contains commented-out example usage.