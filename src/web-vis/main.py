# backend.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
from umap import UMAP
import sys
import os

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helper_utils import load_embeddings_from_db
from reduce_dimension_storage import get_reduced_embeddings

# Pre-computed projections cache
PRECOMPUTED_PROJECTIONS = None

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# Data ---
EMBEDDIG_DIM = 1024
EMBEDDINGS_DATA = load_embeddings_from_db("../data/db.db") # names:tuple, data:ndarray of shape: N x EMBEDDIG_DIM
print(f"✅ Data has been loaded Successfully. shape: {EMBEDDINGS_DATA[1].shape}")

# Try to load pre-computed projections at startup
try:
    names, projections_array = get_reduced_embeddings("../data/db.db", dims=3)
    if len(names) > 0:
        PRECOMPUTED_PROJECTIONS = [{"name": name, "x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
                                  for name, p in zip(names, projections_array)]
        print(f"✅ Pre-computed projections loaded successfully for {len(names)} documents")
    else:
        print("⚠️ No pre-computed projections found in database, UMAP will be computed dynamically")
except Exception as e:
    print(f"⚠️ Error loading pre-computed projections: {e}. UMAP will be computed dynamically") 

def compute_umap(data: tuple, n_neighbors: int = 4, min_dist: float = 0.1, metric: str = "euclidean") -> dict:
    names, vectors = data[0], data[1]

    # removed random_state=42 for parallelism, otherwise you'll get this warningUserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
    reducer = UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric) 
    projections = reducer.fit_transform(vectors)
    
    # Normalize to [-1, 1] range
    projections = (projections - projections.min(axis=0)) / (projections.max(axis=0) - projections.min(axis=0)) * 2 - 1
    
    return [{"name": name, "x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for name, p in zip(names, projections)]

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("opus_45_index.html", {"request": request})


@app.post("/api/umap", response_class=JSONResponse)
async def umap_projection(request: Request):
    form = await request.form()
    n_neighbors = int(form.get("n_neighbors", 15))
    min_dist = float(form.get("min_dist", 0.1))
    metric = form.get("metric", "euclidean")
    print(f"Re-Projecting data for shape {EMBEDDINGS_DATA[1].shape} using UMAP: {n_neighbors, min_dist, metric=}")
    points = compute_umap(EMBEDDINGS_DATA, n_neighbors, min_dist, metric)
    print(f"✅ Data Re-Projection completed")
    return JSONResponse(content=points)

@app.get("/api/umap", response_class=JSONResponse)
async def umap_default():
    # Use pre-computed projections if available, otherwise compute on demand
    if PRECOMPUTED_PROJECTIONS is not None:
        print(f"✅ Returning pre-computed projections for {len(PRECOMPUTED_PROJECTIONS)} documents")
        return JSONResponse(content=PRECOMPUTED_PROJECTIONS)
    else:
        print(f"Projecting data for shape {EMBEDDINGS_DATA[1].shape} using UMAP (default parameters)")
        points = compute_umap(EMBEDDINGS_DATA)
        print(f"✅ Data projection completed")
        return JSONResponse(content=points)
