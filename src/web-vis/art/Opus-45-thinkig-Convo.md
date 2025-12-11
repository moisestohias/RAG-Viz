
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
EMBEDDINGS_DATA = load_embeddings_from_db("../data/db.db") # files, data: data shape: N x EMBEDDIG_DIM

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
===
# Frontend Implementation

## File Structure
```
├── main.py
├── static/
│   └── js/
│       └── app.js
└── templates/
    └── index.html
```

## templates/index.html

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UMAP Embeddings</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/htmx.org@1.9.10"></script>
    <script src="https://unpkg.com/hyperscript.org@0.9.12"></script>
    <script src="https://d3js.org/d3.v7.min.js"></script>
</head>
<body class="bg-gray-900 text-white min-h-screen">
    <header class="p-4 bg-gray-800 border-b border-gray-700">
        <form hx-post="/api/umap" hx-swap="none"
              _="on htmx:beforeRequest add .opacity-40 to #viz
                 on htmx:afterRequest remove .opacity-40 from #viz then call updateViz(event.detail.xhr.responseText)">
            <div class="flex gap-4 items-end flex-wrap">
                <div>
                    <label class="block text-xs text-gray-400 mb-1">n_neighbors</label>
                    <input type="number" name="n_neighbors" value="15" min="2" max="100"
                           class="bg-gray-700 border border-gray-600 rounded px-2 py-1 w-24 text-sm focus:outline-none focus:border-blue-500">
                </div>
                <div>
                    <label class="block text-xs text-gray-400 mb-1">min_dist</label>
                    <input type="number" name="min_dist" value="0.1" min="0" max="1" step="0.05"
                           class="bg-gray-700 border border-gray-600 rounded px-2 py-1 w-24 text-sm focus:outline-none focus:border-blue-500">
                </div>
                <div>
                    <label class="block text-xs text-gray-400 mb-1">metric</label>
                    <select name="metric" class="bg-gray-700 border border-gray-600 rounded px-2 py-1 text-sm focus:outline-none focus:border-blue-500">
                        <option value="euclidean">euclidean</option>
                        <option value="cosine">cosine</option>
                        <option value="manhattan">manhattan</option>
                    </select>
                </div>
                <button type="submit" class="bg-blue-600 hover:bg-blue-500 px-4 py-1.5 rounded text-sm transition-colors">
                    Compute
                </button>
                <span class="text-xs text-gray-500 self-center ml-auto">drag to rotate</span>
            </div>
        </form>
    </header>

    <main class="flex justify-center items-center" style="height:calc(100vh - 72px)">
        <svg id="viz" class="transition-opacity duration-300 cursor-grab active:cursor-grabbing"></svg>
    </main>

    <div id="tooltip" class="fixed bg-black/90 border border-gray-700 px-2 py-1 rounded text-xs pointer-events-none opacity-0 transition-opacity z-50"></div>

    <script src="/static/js/app.js"></script>
</body>
</html>
```

## static/js/app.js

```javascript
const W = 900, H = 650;
const svg = d3.select("#viz").attr("width", W).attr("height", H);
const tip = d3.select("#tooltip");
const color = d3.scaleSequential(d3.interpolateViridis);

let pts = [], ax = 0.4, ay = -0.5;

const project = p => {
    const cy = Math.cos(ay), sy = Math.sin(ay), cx = Math.cos(ax), sx = Math.sin(ax);
    const x1 = p.x * cy - p.z * sy, z1 = p.x * sy + p.z * cy;
    const y1 = p.y * cx - z1 * sx, z2 = p.y * sx + z1 * cx;
    const s = 180 / (4 + z2);
    return { px: x1 * s + W / 2, py: y1 * s + H / 2, pz: z2, name: p.name, idx: p.idx };
};

const normalize = data => {
    const extent = key => d3.extent(data, d => d[key]);
    const scale = key => d3.scaleLinear().domain(extent(key)).range([-2, 2]);
    const [sx, sy, sz] = ['x', 'y', 'z'].map(scale);
    return data.map((d, i) => ({ name: d.name, x: sx(d.x), y: sy(d.y), z: sz(d.z), idx: i / data.length }));
};

const render = () => {
    const projected = pts.map(project).sort((a, b) => a.pz - b.pz);
    svg.selectAll("circle")
        .data(projected, d => d.name)
        .join("circle")
        .attr("cx", d => d.px)
        .attr("cy", d => d.py)
        .attr("r", d => 4 + (d.pz + 2) * 1.5)
        .attr("fill", d => color(d.idx))
        .attr("opacity", d => 0.4 + 0.5 * (d.pz + 2) / 4)
        .on("mouseenter", (e, d) => tip.style("opacity", 1).text(d.name))
        .on("mousemove", e => tip.style("left", `${e.pageX + 12}px`).style("top", `${e.pageY - 20}px`))
        .on("mouseleave", () => tip.style("opacity", 0));
};

window.updateViz = res => { pts = normalize(JSON.parse(res)); render(); };

svg.call(d3.drag().on("drag", e => { ay += e.dx * 0.008; ax += e.dy * 0.008; render(); }));

fetch("/api/umap").then(r => r.json()).then(d => { pts = normalize(d); render(); });
```

---

## How It Works

| Component | Role |
|-----------|------|
| **htmx** | Handles form POST to `/api/umap` without page reload |
| **hyperscript** | Adds loading state, calls `updateViz()` on response |
| **D3.js** | Renders 3D scatter plot with perspective projection |
| **TailwindCSS** | Minimal dark-theme styling |

**Interaction flow:**
1. Page loads → fetches default UMAP projection → renders 3D plot
2. User adjusts parameters → submits form → htmx POSTs to backend
3. Backend recomputes UMAP → returns JSON
4. Hyperscript calls `updateViz()` → D3 re-renders with new data
5. Drag on canvas rotates the 3D view

=== 
> Q: add a the ability too zoom in an out

## static/js/app.js

```javascript
const W = 900, H = 650;
const svg = d3.select("#viz").attr("width", W).attr("height", H);
const tip = d3.select("#tooltip");
const color = d3.scaleSequential(d3.interpolateViridis);

let pts = [], ax = 0.4, ay = -0.5, scale = 1;

const project = p => {
    const cy = Math.cos(ay), sy = Math.sin(ay), cx = Math.cos(ax), sx = Math.sin(ax);
    const x1 = p.x * cy - p.z * sy, z1 = p.x * sy + p.z * cy;
    const y1 = p.y * cx - z1 * sx, z2 = p.y * sx + z1 * cx;
    const s = (180 * scale) / (4 + z2);
    return { px: x1 * s + W / 2, py: y1 * s + H / 2, pz: z2, name: p.name, idx: p.idx };
};

const normalize = data => {
    const extent = key => d3.extent(data, d => d[key]);
    const sc = key => d3.scaleLinear().domain(extent(key)).range([-2, 2]);
    const [sx, sy, sz] = ['x', 'y', 'z'].map(sc);
    return data.map((d, i) => ({ name: d.name, x: sx(d.x), y: sy(d.y), z: sz(d.z), idx: i / data.length }));
};

const render = () => {
    const projected = pts.map(project).sort((a, b) => a.pz - b.pz);
    svg.selectAll("circle")
        .data(projected, d => d.name)
        .join("circle")
        .attr("cx", d => d.px)
        .attr("cy", d => d.py)
        .attr("r", d => Math.max(2, (4 + (d.pz + 2) * 1.5) * Math.sqrt(scale)))
        .attr("fill", d => color(d.idx))
        .attr("opacity", d => 0.4 + 0.5 * (d.pz + 2) / 4)
        .on("mouseenter", (e, d) => tip.style("opacity", 1).text(d.name))
        .on("mousemove", e => tip.style("left", `${e.pageX + 12}px`).style("top", `${e.pageY - 20}px`))
        .on("mouseleave", () => tip.style("opacity", 0));
};

window.updateViz = res => { pts = normalize(JSON.parse(res)); render(); };

svg.call(d3.drag().on("drag", e => { ay += e.dx * 0.008; ax += e.dy * 0.008; render(); }));

svg.call(d3.zoom()
    .scaleExtent([0.3, 10])
    .filter(e => e.type === 'wheel' || e.type === 'dblclick')
    .on("zoom", e => { scale = e.transform.k; render(); }));

fetch("/api/umap").then(r => r.json()).then(d => { pts = normalize(d); render(); });
```

---

## Changes Made

| Change | Description |
|--------|-------------|
| `scale` variable | Tracks current zoom level (default: 1) |
| Projection update | Multiplies perspective by `scale` |
| Circle radius | Scales with `Math.sqrt(scale)` for balanced sizing |
| `d3.zoom()` | Handles scroll wheel and double-click zoom |
| `.filter()` | Separates zoom (wheel) from drag (mouse move) |
| `.scaleExtent([0.3, 10])` | Limits zoom range |

**Controls:**
- **Drag** → rotate 3D view
- **Scroll wheel** → zoom in/out
- **Double-click** → zoom in

===
> Q: How to control the size of the dots (points) on the graph?

Add a **point size slider** that updates in real-time (client-side only, no backend call needed).

## templates/index.html

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UMAP Embeddings</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/htmx.org@1.9.10"></script>
    <script src="https://unpkg.com/hyperscript.org@0.9.12"></script>
    <script src="https://d3js.org/d3.v7.min.js"></script>
</head>
<body class="bg-gray-900 text-white min-h-screen">
    <header class="p-4 bg-gray-800 border-b border-gray-700">
        <form hx-post="/api/umap" hx-swap="none"
              _="on htmx:beforeRequest add .opacity-40 to #viz
                 on htmx:afterRequest remove .opacity-40 from #viz then call updateViz(event.detail.xhr.responseText)">
            <div class="flex gap-4 items-end flex-wrap">
                <div>
                    <label class="block text-xs text-gray-400 mb-1">n_neighbors</label>
                    <input type="number" name="n_neighbors" value="15" min="2" max="100"
                           class="bg-gray-700 border border-gray-600 rounded px-2 py-1 w-24 text-sm focus:outline-none focus:border-blue-500">
                </div>
                <div>
                    <label class="block text-xs text-gray-400 mb-1">min_dist</label>
                    <input type="number" name="min_dist" value="0.1" min="0" max="1" step="0.05"
                           class="bg-gray-700 border border-gray-600 rounded px-2 py-1 w-24 text-sm focus:outline-none focus:border-blue-500">
                </div>
                <div>
                    <label class="block text-xs text-gray-400 mb-1">metric</label>
                    <select name="metric" class="bg-gray-700 border border-gray-600 rounded px-2 py-1 text-sm focus:outline-none focus:border-blue-500">
                        <option value="euclidean">euclidean</option>
                        <option value="cosine">cosine</option>
                        <option value="manhattan">manhattan</option>
                    </select>
                </div>
                <div class="border-l border-gray-600 pl-4">
                    <label class="block text-xs text-gray-400 mb-1">point size: <span id="size-val">4</span></label>
                    <input type="range" id="point-size" min="1" max="20" value="4" step="0.5"
                           class="w-24 accent-blue-500"
                           _="on input set #size-val.textContent to my.value then call setPointSize(my.value)">
                </div>
                <button type="submit" class="bg-blue-600 hover:bg-blue-500 px-4 py-1.5 rounded text-sm transition-colors">
                    Compute
                </button>
                <span class="text-xs text-gray-500 self-center ml-auto">drag to rotate • scroll to zoom</span>
            </div>
        </form>
    </header>

    <main class="flex justify-center items-center" style="height:calc(100vh - 72px)">
        <svg id="viz" class="transition-opacity duration-300 cursor-grab active:cursor-grabbing"></svg>
    </main>

    <div id="tooltip" class="fixed bg-black/90 border border-gray-700 px-2 py-1 rounded text-xs pointer-events-none opacity-0 transition-opacity z-50"></div>

    <script src="/static/js/app.js"></script>
</body>
</html>
```

## static/js/app.js

```javascript
const W = 900, H = 650;
const svg = d3.select("#viz").attr("width", W).attr("height", H);
const tip = d3.select("#tooltip");
const color = d3.scaleSequential(d3.interpolateViridis);

let pts = [], ax = 0.4, ay = -0.5, scale = 1, baseSize = 4;

const project = p => {
    const cy = Math.cos(ay), sy = Math.sin(ay), cx = Math.cos(ax), sx = Math.sin(ax);
    const x1 = p.x * cy - p.z * sy, z1 = p.x * sy + p.z * cy;
    const y1 = p.y * cx - z1 * sx, z2 = p.y * sx + z1 * cx;
    const s = (180 * scale) / (4 + z2);
    return { px: x1 * s + W / 2, py: y1 * s + H / 2, pz: z2, name: p.name, idx: p.idx };
};

const normalize = data => {
    const extent = key => d3.extent(data, d => d[key]);
    const sc = key => d3.scaleLinear().domain(extent(key)).range([-2, 2]);
    const [sx, sy, sz] = ['x', 'y', 'z'].map(sc);
    return data.map((d, i) => ({ name: d.name, x: sx(d.x), y: sy(d.y), z: sz(d.z), idx: i / data.length }));
};

const render = () => {
    const projected = pts.map(project).sort((a, b) => a.pz - b.pz);
    svg.selectAll("circle")
        .data(projected, d => d.name)
        .join("circle")
        .attr("cx", d => d.px)
        .attr("cy", d => d.py)
        .attr("r", d => Math.max(1, (baseSize + (d.pz + 2) * (baseSize * 0.3)) * Math.sqrt(scale)))
        .attr("fill", d => color(d.idx))
        .attr("opacity", d => 0.4 + 0.5 * (d.pz + 2) / 4)
        .on("mouseenter", (e, d) => tip.style("opacity", 1).text(d.name))
        .on("mousemove", e => tip.style("left", `${e.pageX + 12}px`).style("top", `${e.pageY - 20}px`))
        .on("mouseleave", () => tip.style("opacity", 0));
};

window.updateViz = res => { pts = normalize(JSON.parse(res)); render(); };
window.setPointSize = val => { baseSize = parseFloat(val); render(); };

svg.call(d3.drag().on("drag", e => { ay += e.dx * 0.008; ax += e.dy * 0.008; render(); }));

svg.call(d3.zoom()
    .scaleExtent([0.3, 10])
    .filter(e => e.type === 'wheel' || e.type === 'dblclick')
    .on("zoom", e => { scale = e.transform.k; render(); }));

fetch("/api/umap").then(r => r.json()).then(d => { pts = normalize(d); render(); });
```

---

## Changes Summary

| Change | Location | Description |
|--------|----------|-------------|
| `baseSize` variable | JS | Controls base point radius (default: 4) |
| `setPointSize()` | JS | Global function to update size and re-render |
| Range slider | HTML | Input control with min=1, max=20 |
| Hyperscript binding | HTML | `_="on input..."` calls `setPointSize()` live |
| Dynamic radius formula | JS | `baseSize + (d.pz + 2) * (baseSize * 0.3)` for depth effect |

**The slider updates points instantly without any backend call.**


===

> Q: Great, let's improve the zoom, my cluster is huge, I would like to be able to zoom so much, also currently the drag experience is not great, not intuitive I would like to be able to zoom in and rotate the data intuitively to understand the data, when drag the graph the graph rotate in weird manner, 

## static/js/app.js

```javascript
const W = 900, H = 650;
const svg = d3.select("#viz").attr("width", W).attr("height", H);
const tip = d3.select("#tooltip");
const color = d3.scaleSequential(d3.interpolateViridis);

let pts = [], scale = 1, baseSize = 4;
let panX = 0, panY = 0;
let rotMat = [[1,0,0],[0,1,0],[0,0,1]];

const multiply = (a, b) => a.map((row, i) => b[0].map((_, j) => row.reduce((sum, _, k) => sum + a[i][k] * b[k][j], 0)));

const rotateX = a => [[1,0,0],[0,Math.cos(a),-Math.sin(a)],[0,Math.sin(a),Math.cos(a)]];
const rotateY = a => [[Math.cos(a),0,Math.sin(a)],[0,1,0],[-Math.sin(a),0,Math.cos(a)]];

const applyRot = (dx, dy) => {
    const rx = rotateX(-dy * 0.005);
    const ry = rotateY(-dx * 0.005);
    rotMat = multiply(ry, multiply(rx, rotMat));
};

const project = p => {
    const [x, y, z] = [
        p.x * rotMat[0][0] + p.y * rotMat[0][1] + p.z * rotMat[0][2],
        p.x * rotMat[1][0] + p.y * rotMat[1][1] + p.z * rotMat[1][2],
        p.x * rotMat[2][0] + p.y * rotMat[2][1] + p.z * rotMat[2][2]
    ];
    const perspective = (200 * scale) / (5 + z);
    return {
        px: x * perspective + W / 2 + panX,
        py: y * perspective + H / 2 + panY,
        pz: z,
        name: p.name,
        idx: p.idx
    };
};

const normalize = data => {
    const extent = key => d3.extent(data, d => d[key]);
    const sc = key => d3.scaleLinear().domain(extent(key)).range([-2, 2]);
    const [sx, sy, sz] = ['x', 'y', 'z'].map(sc);
    return data.map((d, i) => ({ name: d.name, x: sx(d.x), y: sy(d.y), z: sz(d.z), idx: i / data.length }));
};

const render = () => {
    const projected = pts.map(project).sort((a, b) => a.pz - b.pz);
    svg.selectAll("circle")
        .data(projected, d => d.name)
        .join("circle")
        .attr("cx", d => d.px)
        .attr("cy", d => d.py)
        .attr("r", d => Math.max(1, (baseSize + (d.pz + 2) * (baseSize * 0.25)) * Math.sqrt(scale)))
        .attr("fill", d => color(d.idx))
        .attr("opacity", d => 0.5 + 0.4 * (d.pz + 2) / 4)
        .on("mouseenter", (e, d) => tip.style("opacity", 1).text(d.name))
        .on("mousemove", e => tip.style("left", `${e.pageX + 12}px`).style("top", `${e.pageY - 20}px`))
        .on("mouseleave", () => tip.style("opacity", 0));
};

window.updateViz = res => { pts = normalize(JSON.parse(res)); render(); };
window.setPointSize = val => { baseSize = parseFloat(val); render(); };
window.resetView = () => { 
    rotMat = [[1,0,0],[0,1,0],[0,0,1]]; 
    scale = 1; panX = 0; panY = 0; 
    render(); 
};

svg.call(d3.drag()
    .on("drag", e => {
        if (e.sourceEvent.shiftKey) {
            panX += e.dx;
            panY += e.dy;
        } else {
            applyRot(e.dx, e.dy);
        }
        render();
    }));

svg.call(d3.zoom()
    .scaleExtent([0.05, 100])
    .filter(e => e.type === 'wheel' || e.type === 'dblclick')
    .on("zoom", e => { scale = e.transform.k; render(); }));

svg.on("contextmenu", e => e.preventDefault());

fetch("/api/umap").then(r => r.json()).then(d => { pts = normalize(d); render(); });
```

## templates/index.html

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UMAP Embeddings</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/htmx.org@1.9.10"></script>
    <script src="https://unpkg.com/hyperscript.org@0.9.12"></script>
    <script src="https://d3js.org/d3.v7.min.js"></script>
</head>
<body class="bg-gray-900 text-white min-h-screen">
    <header class="p-4 bg-gray-800 border-b border-gray-700">
        <form hx-post="/api/umap" hx-swap="none"
              _="on htmx:beforeRequest add .opacity-40 to #viz
                 on htmx:afterRequest remove .opacity-40 from #viz then call updateViz(event.detail.xhr.responseText)">
            <div class="flex gap-4 items-end flex-wrap">
                <div>
                    <label class="block text-xs text-gray-400 mb-1">n_neighbors</label>
                    <input type="number" name="n_neighbors" value="15" min="2" max="100"
                           class="bg-gray-700 border border-gray-600 rounded px-2 py-1 w-24 text-sm focus:outline-none focus:border-blue-500">
                </div>
                <div>
                    <label class="block text-xs text-gray-400 mb-1">min_dist</label>
                    <input type="number" name="min_dist" value="0.1" min="0" max="1" step="0.05"
                           class="bg-gray-700 border border-gray-600 rounded px-2 py-1 w-24 text-sm focus:outline-none focus:border-blue-500">
                </div>
                <div>
                    <label class="block text-xs text-gray-400 mb-1">metric</label>
                    <select name="metric" class="bg-gray-700 border border-gray-600 rounded px-2 py-1 text-sm focus:outline-none focus:border-blue-500">
                        <option value="euclidean">euclidean</option>
                        <option value="cosine">cosine</option>
                        <option value="manhattan">manhattan</option>
                    </select>
                </div>
                <div class="border-l border-gray-600 pl-4">
                    <label class="block text-xs text-gray-400 mb-1">point size: <span id="size-val">4</span></label>
                    <input type="range" id="point-size" min="1" max="20" value="4" step="0.5"
                           class="w-24 accent-blue-500"
                           _="on input set #size-val.textContent to my.value then call setPointSize(my.value)">
                </div>
                <button type="submit" class="bg-blue-600 hover:bg-blue-500 px-4 py-1.5 rounded text-sm transition-colors">
                    Compute
                </button>
                <button type="button" class="bg-gray-600 hover:bg-gray-500 px-3 py-1.5 rounded text-sm transition-colors"
                        _="on click call resetView()">
                    Reset
                </button>
                <div class="text-xs text-gray-500 self-center ml-auto text-right leading-relaxed">
                    <div>drag: rotate • scroll: zoom</div>
                    <div>shift+drag: pan</div>
                </div>
            </div>
        </form>
    </header>

    <main class="flex justify-center items-center" style="height:calc(100vh - 72px)">
        <svg id="viz" class="transition-opacity duration-300 cursor-grab active:cursor-grabbing"></svg>
    </main>

    <div id="tooltip" class="fixed bg-black/90 border border-gray-700 px-2 py-1 rounded text-xs pointer-events-none opacity-0 transition-opacity z-50"></div>

    <script src="/static/js/app.js"></script>
</body>
</html>
```

---

## Improvements Made

| Issue | Fix |
|-------|-----|
| **Weird rotation** | Uses rotation matrix multiplication — rotates around screen axes, not world axes |
| **Limited zoom** | Extended range: `0.05` to `100` (was 0.3–10) |
| **No panning** | `Shift + drag` to pan around |
| **No reset** | Added "Reset" button to restore default view |

## How Rotation Now Works

```
Before (Euler angles):
  - Accumulated X/Y angles applied in fixed order
  - Felt disconnected from drag direction

After (Rotation matrix):
  - Each drag applies incremental rotation relative to current view
  - Drag left → scene rotates left (from your perspective)
  - Drag up → scene rotates up (from your perspective)
```

## Controls

| Action | Effect |
|--------|--------|
| **Drag** | Rotate (trackball-style, intuitive) |
| **Shift + Drag** | Pan the view |
| **Scroll** | Zoom in/out (0.05x – 100x) |
| **Double-click** | Zoom in |
| **Reset button** | Restore original view |