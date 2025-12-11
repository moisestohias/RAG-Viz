# Implementing "Set Point as Center" (Anchor Point) in D3.js 3D Visualization

## Overview

This document describes how to implement an **anchor point** feature that allows users to click on any point in a 3D visualization to make it the **center of rotation**. All other points will then orbit around this selected anchor when the user rotates the view.

---

## The Problem

In a typical 3D visualization, rotation happens around the **origin (0, 0, 0)**. This works fine initially, but when exploring data:

- Users often find an interesting point and want to examine its neighbors
- With origin-centered rotation, that point moves unpredictably when rotating
- It's hard to keep a point of interest in view while exploring around it

---

## The Solution: Anchor Point

Allow users to **click any point** to make it the rotation center. The selected point stays fixed in the center of the screen while all other points rotate around it.

---

## Implementation Steps

### Step 1: Add Anchor State Variables

Store the anchor point's 3D coordinates and its name (for visual highlighting):

```javascript
let anchor = { x: 0, y: 0, z: 0 };  // 3D coordinates of rotation center
let anchorName = null;              // Name of anchor point (for highlighting)
```

### Step 2: Modify the Projection Function

The key insight is the **translate-rotate-translate** pattern:

1. **Translate** all points so the anchor moves to origin
2. **Rotate** around origin (which is now the anchor)
3. **Project** to screen coordinates

```javascript
const project = p => {
    // Step 1: Translate so anchor is at origin
    const dx = p.x - anchor.x;
    const dy = p.y - anchor.y;
    const dz = p.z - anchor.z;
    
    // Step 2: Apply rotation (around origin, which is now the anchor)
    const cy = Math.cos(ay), sy = Math.sin(ay);
    const cx = Math.cos(ax), sx = Math.sin(ax);
    
    const x1 = dx * cy - dz * sy;
    const z1 = dx * sy + dz * cy;
    const y1 = dy * cx - z1 * sx;
    const z2 = dy * sx + z1 * cx;
    
    // Step 3: Perspective projection
    // Anchor (now at origin after translation) projects to screen center
    const s = (180 * scale) / (4 + z2);
    return { 
        px: x1 * s + W / 2 + panX, 
        py: y1 * s + H / 2 + panY, 
        pz: z2, 
        name: p.name, 
        idx: p.idx 
    };
};
```

**Note:** We do NOT translate back after rotation. This is intentional—the anchor should appear at the screen center, which it does because after translation it's at origin.

### Step 3: Add Click Handler to Set Anchor

When a point is clicked, find its original 3D coordinates and set as anchor:

```javascript
.on("click", (e, d) => {
    e.stopPropagation();  // Prevent SVG background click
    
    // Find original point with 3D coordinates
    const originalPoint = pts.find(p => p.name === d.name);
    
    if (originalPoint) {
        anchor = { 
            x: originalPoint.x, 
            y: originalPoint.y, 
            z: originalPoint.z 
        };
        anchorName = d.name;
        render();
    }
});
```

**Important:** We need to look up the original point data because the projected data (`d`) only has screen coordinates (`px`, `py`), not the 3D coordinates we need.

### Step 4: Add Visual Feedback

Highlight the anchor point so users know which point is selected:

```javascript
// In the render function
.attr("stroke", d => d.name === anchorName ? "#ffffff" : "none")
.attr("stroke-width", d => d.name === anchorName ? 2 : 0)
.style("cursor", "pointer")

// Enhanced tooltip
.on("mouseenter", (e, d) => {
    const label = d.name + (d.name === anchorName ? " (anchor)" : "");
    tip.style("opacity", 1).text(label);
})
```

### Step 5: Add Reset Functionality

Allow users to reset to origin-centered rotation:

```javascript
// Double-click on SVG background to reset
svg.on("dblclick", (e) => {
    // Only reset if clicking on background, not on a point
    if (e.target.tagName === 'svg') {
        anchor = { x: 0, y: 0, z: 0 };
        anchorName = null;
        render();
    }
});
```

---

## Complete Implementation

```javascript
// State
let anchor = { x: 0, y: 0, z: 0 }, anchorName = null;

// Projection with anchor support
const project = p => {
    const dx = p.x - anchor.x;
    const dy = p.y - anchor.y;
    const dz = p.z - anchor.z;
    
    const cy = Math.cos(ay), sy = Math.sin(ay);
    const cx = Math.cos(ax), sx = Math.sin(ax);
    const x1 = dx * cy - dz * sy, z1 = dx * sy + dz * cy;
    const y1 = dy * cx - z1 * sx, z2 = dy * sx + z1 * cx;
    
    const s = (180 * scale) / (4 + z2);
    return { px: x1 * s + W / 2 + panX, py: y1 * s + H / 2 + panY, pz: z2, name: p.name, idx: p.idx };
};

// Render with click handler and visual feedback
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
            e.stopPropagation();
            const originalPoint = pts.find(p => p.name === d.name);
            if (originalPoint) {
                anchor = { x: originalPoint.x, y: originalPoint.y, z: originalPoint.z };
                anchorName = d.name;
                render();
            }
        });
};

// Reset anchor on background double-click
svg.on("dblclick", (e) => {
    if (e.target.tagName === 'svg') {
        anchor = { x: 0, y: 0, z: 0 };
        anchorName = null;
        render();
    }
});
```

---

## The Math: Why This Works

### Standard Rotation (around origin)

When you rotate a point `P` around the origin:
```
P' = R × P
```
Where `R` is the rotation matrix.

### Rotation Around Arbitrary Point A

To rotate around point `A` instead of origin:
```
P' = R × (P - A) + A
```

But we **want the anchor to appear at screen center**, so we skip the final `+ A`:
```
P' = R × (P - A)
```

This means:
- The anchor point becomes `R × (A - A) = R × 0 = 0` → projects to center
- All other points rotate around where the anchor was

---

## User Interaction Flow

| Action | Result |
|--------|--------|
| **Click point** | That point becomes anchor, highlighted with white ring |
| **Drag** | Rotation happens around anchor (anchor stays centered) |
| **Double-click background** | Reset to origin-centered rotation |
| **Hover anchor** | Tooltip shows "(anchor)" suffix |

---

## Key Takeaways

1. **Translate-Rotate pattern** - Classic technique for rotating around arbitrary points
2. **Don't translate back** - Keep anchor at origin for centered projection
3. **Use original 3D coords** - Click handler receives projected data, must look up original
4. **Visual feedback is essential** - Users need to know which point is the anchor
5. **Provide reset mechanism** - Users should be able to return to default behavior
