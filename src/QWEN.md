# Vault Organizer - RAG Project

## Project Overview
Vault Organizer is a Retrieval-Augmented Generation (RAG) system designed to organize, index, and search through a collection of documents (primarily Markdown files) using vector embeddings. The project leverages SQLite with the sqlite-vec extension to create and store vector embeddings for efficient similarity search. The system extracts content snippets from Markdown files, generates embeddings using local AI models (specifically Qwen3-Embedding), and stores them in a vector database for semantic search capabilities.

### Main Components
- **embedder.py**: Contains logic for generating text embeddings using a local API endpoint (Ollama-style), with retry mechanisms and timing decorators. Supports different task prompts (clustering, retrieval) for the Qwen3-Embedding model.
- **indexer.py**: Scans directories for Markdown (.md) files, extracts content snippets (first 200 words, excluding base64 image data), and creates an indexed.json file mapping file paths to their content.
- **populate_sqlite_vec_db.py**: Reads the indexed data, computes vector embeddings, and populates an SQLite database with both the embeddings and raw text for efficient search.
- **helper_utils.py**: Utility functions for path expansion and file validation.

## Building and Running

### Prerequisites
- Python 3.12+
- Ollama with the Qwen3-Embedding model installed (`dengcao/Qwen3-Embedding-0.6B:Q8_0`)
- SQLite with sqlite-vec extension installed (`~/.local/vec0.so`)
- Dependencies installed via `uv sync` or `pip install`

### Setup
1. Install dependencies:
```bash
uv sync  # or pip install -r requirements.txt if using pip
```

2. Ensure Ollama is running and the required model is downloaded:
```bash
ollama pull dengcao/Qwen3-Embedding-0.6B:Q8_0
ollama serve  # in a separate terminal
```

3. Make sure sqlite-vec extension is properly installed and accessible at `~/.local/vec0.so`

### Running the System
1. Index Markdown files to create indexed.json:
```bash
cd src
python indexer.py
```

2. Generate embeddings and populate the SQLite database:
```bash
python populate_sqlite_vec_db.py
```

The system will create/update the `data/db.db` SQLite database with vector embeddings and text content.

## Development Conventions
- The code follows Python 3.12+ syntax and conventions
- Error handling includes retries for API calls and proper file validation
- Data processing includes filtering of base64 image data and YAML frontmatter parsing
- The system saves progress incrementally to prevent data loss during long processing runs
- Embedding dimensions are hardcoded (currently 1024) to match the Qwen3-Embedding model
- The code includes timing decorators to monitor performance
