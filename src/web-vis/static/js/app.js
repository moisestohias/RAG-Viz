// const W = 900, H = 650;
const W = 1920, H = 900;
const svg = d3.select("#viz").attr("width", W).attr("height", H);
const tip = d3.select("#tooltip");
const color = d3.scaleSequential(d3.interpolateViridis);

let pts = [], ax = 0.4, ay = -0.5, scale = 3, baseSize = 1, panX = 0, panY = 0, ctrlPressed = false, prevScale = 3;
let anchor = { x: 0, y: 0, z: 0 }, anchorName = null;  // Anchor point for rotation center

// Track CTRL key globally for reliable detection
document.addEventListener('keydown', e => { if (e.key === 'Control') ctrlPressed = true; });
document.addEventListener('keyup', e => { if (e.key === 'Control') ctrlPressed = false; });
window.addEventListener('blur', () => { ctrlPressed = false; }); // Reset when window loses focus

const project = p => {
    // Translate so anchor is at origin (rotate around anchor)
    const dx = p.x - anchor.x;
    const dy = p.y - anchor.y;
    const dz = p.z - anchor.z;

    // Apply rotation
    const cy = Math.cos(ay), sy = Math.sin(ay), cx = Math.cos(ax), sx = Math.sin(ax);
    const x1 = dx * cy - dz * sy, z1 = dx * sy + dz * cy;
    const y1 = dy * cx - z1 * sx, z2 = dy * sx + z1 * cx;

    // Perspective projection (anchor stays at center)
    const s = (180 * scale) / (4 + z2);
    return { px: x1 * s + W / 2 + panX, py: y1 * s + H / 2 + panY, pz: z2, name: p.name, idx: p.idx };
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
        .attr("stroke", d => d.name === anchorName ? "#ffffff" : "none")
        .attr("stroke-width", d => d.name === anchorName ? 2 : 0)
        .style("cursor", "pointer")
        .on("mouseenter", (e, d) => tip.style("opacity", 1).text(d.name + (d.name === anchorName ? " (anchor)" : "")))
        .on("mousemove", e => tip.style("left", `${e.pageX + 12}px`).style("top", `${e.pageY - 20}px`))
        .on("mouseleave", () => tip.style("opacity", 0))
        .on("click", (e, d) => {
            e.stopPropagation();  // Prevent SVG click handler
            // Find original point data with 3D coordinates
            const originalPoint = pts.find(p => p.name === d.name);
            if (originalPoint) {
                anchor = { x: originalPoint.x, y: originalPoint.y, z: originalPoint.z };
                anchorName = d.name;
                render();
            }
        });
};

window.updateViz = res => { pts = normalize(JSON.parse(res)); render(); };
window.setPointSize = val => { baseSize = parseFloat(val); render(); };

// Double-click on background to reset anchor to origin
svg.on("dblclick", (e) => {
    // Only reset if clicking on background, not on a point
    if (e.target.tagName === 'svg') {
        anchor = { x: 0, y: 0, z: 0 };
        anchorName = null;
        render();
    }
});

svg.call(d3.drag()
    .filter(e => !e.button) // Allow drag with any modifier key (CTRL, Shift, etc.)
    .on("drag", e => {
        if (ctrlPressed) {
            // CTRL+drag: pan the visualization
            panX += e.dx;
            panY += e.dy;
        } else {
            // Normal drag: rotate the visualization
            ay += e.dx * 0.008;
            ax += e.dy * 0.008;
        }
        render();
    }));

svg.call(d3.zoom()
    .scaleExtent([3, 300])
    .filter(e => e.type === 'wheel' || e.type === 'dblclick')
    .on("zoom", (e) => {
        const newScale = e.transform.k;
        const scaleRatio = newScale / prevScale;

        // Get mouse position relative to SVG
        const [mouseX, mouseY] = d3.pointer(e, svg.node());

        // The visualization center (before pan) is at (W/2, H/2)
        // Current visual center with pan is at (W/2 + panX, H/2 + panY)
        // Mouse offset from visual center
        const offsetX = mouseX - (W / 2 + panX);
        const offsetY = mouseY - (H / 2 + panY);

        // When scaling, points move away from center by scaleRatio
        // Adjust pan to keep the point under mouse fixed
        panX += offsetX - offsetX * scaleRatio;
        panY += offsetY - offsetY * scaleRatio;

        prevScale = newScale;
        scale = newScale;
        render();
    }));

fetch("/api/umap").then(r => r.json()).then(d => { pts = normalize(d); render(); });