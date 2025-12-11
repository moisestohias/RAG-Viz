# Web Visualization for RAG Project - UMAP Embeddings Visualizer

## Project Overview

This is a web-based visualization tool for the RAG (Retrieval Augmented Generation) project that allows users to visualize document embeddings using UMAP dimensionality reduction. The application uses FastAPI as a backend to serve a 3D interactive visualization of document embeddings, allowing users to explore relationships between documents in a reduced dimensional space.

### Main Components
- **main.py**: FastAPI backend that loads embeddings from an SQLite database, performs UMAP dimensionality reduction, and serves the frontend application.
- **Templates (HTML)**: Frontend interface using Tailwind CSS, D3.js, HTMX, and hyperscript for interactive visualization.
- **Static Files (JS)**: Client-side JavaScript using D3.js for 3D rendering of embeddings as an interactive scatter plot.
- **Database Integration**: Works with SQLite database containing document embeddings (from the main RAG project).
- **helper_utils.py**: Utilities for loading embeddings from the SQLite database with the sqlite-vec extension.
- **reduce_dimension_storage.py**: Module for pre-computing and storing reduced-dimensional embeddings in the database for faster loading.

## Technologies Used

- **Backend**: FastAPI (Python)
- **Frontend**: HTML, JavaScript (D3.js for visualization)
- **Styling**: Tailwind CSS
- **Interactivity**: HTMX, hyperscript
- **Dimensionality Reduction**: UMAP (Uniform Manifold Approximation and Projection)
- **Data Processing**: NumPy
- **Database**: SQLite with sqlite-vec extension

## Architecture

The application consists of two main parts:

1. **Backend (main.py)**:
   - Loads embeddings from SQLite database using `helper_utils.py`
   - Provides API endpoints for requesting UMAP projections
   - Offers dynamic computation of projections with adjustable UMAP parameters
   - Caches pre-computed projections when available

2. **Frontend (HTML/CSS/JS)**:
   - Interactive 3D visualization of embeddings
   - Adjustable controls for UMAP parameters (n_neighbors, min_dist, metric)
   - Drag to rotate, scroll to zoom functionality
   - Tooltip showing document names on hover

## Building and Running

### Prerequisites
- Python 3.12+
- FastAPI and Uvicorn
- UMAP for dimensionality reduction
- SQLite with sqlite-vec extension installed (`~/.local/vec0.so`)
- A populated SQLite database with embeddings from the main RAG project

### Setup
1. Ensure the main RAG project is set up and the database contains embeddings.

2. Install Python dependencies:
```bash
pip install fastapi uvicorn numpy umap-learn jinja2
```

3. Make sure the database file `../data/db.db` exists and contains embeddings.

### Running the Application
1. From the `/src/web-vis` directory, run:
```bash
uvicorn main:app --reload
```

2. Open `http://localhost:8000` in a web browser to view the visualization.

The application will load pre-computed embeddings from the database on startup and display them in the 3D visualization. Users can adjust UMAP parameters and recompute projections in real-time.

## Development Conventions

- The application follows standard FastAPI project structure
- Error handling and logging are implemented for database operations and UMAP computations
- The code includes pre-computation caching for performance optimization
- Client-side rendering uses D3.js best practices for interactive visualizations
- Parameter validation is performed on both client and server sides