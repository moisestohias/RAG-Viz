## 2025-12-11 23:10
1. Created a comprehensive guide (Saving-Reduced-Embedding.md) detailing different approaches for storing reduced-dimensional embeddings in SQLite with sqlite-vec
2. Implemented the second approach by creating a new module (reduce_dimension_storage.py) that stores UMAP projections in a separate SQLite table
3. Fixed the foreign key constraint issue that was causing errors with sqlite-vec virtual tables
4. Updated the FastAPI web visualization app (web-vis/main.py) to load pre-computed projections at startup while maintaining the ability to dynamically recompute embeddings with custom parameters
5. Fixed the import issue in the FastAPI app by using absolute imports instead of relative imports

The system now supports both fast access to pre-computed 3D projections and the ability to explore data with different UMAP parameters through dynamic computation, providing the best of both worlds for your visualization needs.