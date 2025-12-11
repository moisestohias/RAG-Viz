I have a collection of document embeddings (from a RAG system) that I want to visualize interactively on the web using UMAP for dimensionality reduction. Example input:

```
data = {
  "file1": [0.2414, 0.6712, ...],
  "file2": [0.2414, 0.6712, ...],
  ...
}
```

Build a simple web application that visualizes these embeddings with **D3.js** in a 3D canvas when I hover over a file it should display the file (node) name. The interface should include controls at the top for adjusting **UMAP parameters**, using **htmx** and **hyperscript** to handle interactivity.

When the user updates the UMAP parameters and submits the form, the backend should recompute the projection and update the visualization accordingly. 

# Guidelines

* Keep both frontend and backend implementations minimal and compact.
* Keep comments to a minimum.

## Frontend

* **D3.js** for visualization
* **htmx** and **hyperscript** for interactivity
* **TailwindCSS** for styling

## Backend

* **FastAPI**
* **UMAP** for dimensionality reduction

## Deliverable
front end; how many files you see fit (one or more).

Here's what I have so far,
```python
# main.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
from umap import UMAP

from helper_utils import load_embeddings_from_db

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# Data ---
EMBEDDIG_DIM = 1024
EMBEDDINGS_DATA = load_embeddings_from_db("../data/db.db") # files:list, data:ndarray of shape: N x EMBEDDIG_DIM

def compute_umap(data: tuple, n_neighbors: int = 4, min_dist: float = 0.1, metric: str = "euclidean") -> dict:
    names, vectors = data[0], data[1] # data normalizeed
    reducer = UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric) 
    projections = reducer.fit_transform(vectors)
    return [{"name": name, "x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for name, p in zip(names, projections)]

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/umap", response_class=JSONResponse)
async def umap_projection(request: Request):
    form = await request.form()
    n_neighbors = int(form.get("n_neighbors", 15))
    min_dist = float(form.get("min_dist", 0.1))
    metric = form.get("metric", "euclidean")
    
    points = compute_umap(EMBEDDINGS_DATA, n_neighbors, min_dist, metric)
    return JSONResponse(content=points)

@app.get("/api/umap", response_class=JSONResponse)
async def umap_default():
    print(f"Projecting data for shape {EMBEDDINGS_DATA[1].shape} using UMAP")
    points = compute_umap(EMBEDDINGS_DATA)
    return JSONResponse(content=points)
```
---

I would like to be able to zoom much more, sinse vizualization is for large set points, also I would like to be able to hold ctrl+drag left/right/up/down to navigate to different regions of the visalization. Create a plan on how to you are going to implmenet this changes. The target files are @templates/opus_45_index.html and @static/js/app.js, write your plane detailing what needs to be changes in to FixZoomNavigationPlan.md

