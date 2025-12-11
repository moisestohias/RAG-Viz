# D3.js Zoom-to-Cursor: Issues and Implementation

## Problem Summary

When implementing zoom in D3.js with a custom projection (like a 3D visualization), the default zoom behavior zooms towards the **center of the SVG** rather than towards the **mouse cursor position**. This causes a disorienting effect where the content "slides away" as you zoom in.

---

## The Core Issue

D3's `d3.zoom()` provides a `transform` object with `k` (scale), `x`, and `y` values. However, when you have a **custom projection function** that handles its own coordinate system (like a 3D projection with separate `scale` and `pan` variables), D3's built-in zoom behavior doesn't automatically adjust for zoom-to-cursor.

### What Goes Wrong

1. **Naive implementation** - Simply updating `scale = e.transform.k` causes all points to scale relative to the visualization center, not the mouse position.

2. **Scale mismatch** - If your `scale` variable starts at a different value than your `scaleExtent` minimum, the first zoom event causes a massive jump.

3. **Missing previous scale tracking** - To calculate the zoom ratio correctly, you need to know the *previous* scale, not just the current one.

---

## Root Causes

### 1. Scale Initialization Mismatch

```javascript
// ❌ BUG: scale starts at 1, but scaleExtent starts at 3
let scale = 1;

svg.call(d3.zoom()
    .scaleExtent([3, 100])  // Zoom starts at 3!
    .on("zoom", e => { 
        scale = e.transform.k;  // First zoom: jumps from 1 to 3
    }));
```

On the first zoom event, `transform.k` could be 3 (the minimum), but `scale` is 1. If you try to calculate a ratio for pan adjustment, you get `3/1 = 3`, causing the visualization to jump wildly.

**Fix:** Initialize `scale` to match `scaleExtent[0]`:

```javascript
let scale = 3;  // Match the scaleExtent minimum
```

### 2. Incorrect Scale Ratio Calculation

```javascript
// ❌ BUG: Using current scale instead of previous scale
.on("zoom", e => {
    const scaleRatio = e.transform.k / scale;  // Wrong after first iteration!
    // ... pan adjustment using scaleRatio ...
    scale = e.transform.k;  // Now scale is updated, but we already used it
});
```

After the first zoom, `scale` is updated. On subsequent zooms, `scaleRatio` is calculated using the already-updated `scale`, which is now equal to the previous `transform.k`, not the actual previous scale.

**Fix:** Track previous scale separately:

```javascript
let scale = 3, prevScale = 3;

.on("zoom", e => {
    const scaleRatio = e.transform.k / prevScale;  // Correct ratio
    // ... pan adjustment ...
    prevScale = e.transform.k;
    scale = e.transform.k;
});
```

### 3. Wrong Pan Formula for Zoom-to-Cursor

The key insight is: when you scale, all points move relative to the visualization center. To keep the point under the mouse cursor fixed, you need to adjust the pan to compensate.

```javascript
// ❌ WRONG: This formula doesn't properly account for the center offset
panX = mouseX - (mouseX - panX) * scaleRatio;
```

```javascript
// ✅ CORRECT: Calculate offset from visual center, then compensate
const offsetX = mouseX - (W / 2 + panX);
const offsetY = mouseY - (H / 2 + panY);

panX += offsetX - offsetX * scaleRatio;
panY += offsetY - offsetY * scaleRatio;
```

---

## The Math Behind Zoom-to-Cursor

Given:
- Mouse position: `(mouseX, mouseY)`
- Visualization center: `(W/2, H/2)`
- Current pan: `(panX, panY)`
- Visual center (with pan): `(W/2 + panX, H/2 + panY)`

When scaling by `scaleRatio`:

1. **Offset from visual center:**
   ```
   offsetX = mouseX - (W/2 + panX)
   offsetY = mouseY - (H/2 + panY)
   ```

2. **After scaling, this offset becomes:**
   ```
   newOffsetX = offsetX * scaleRatio
   newOffsetY = offsetY * scaleRatio
   ```

3. **To keep mouse position fixed, adjust pan by the difference:**
   ```
   panX += offsetX - newOffsetX
   panY += offsetY - newOffsetY
   ```
   
   Which simplifies to:
   ```
   panX += offsetX * (1 - scaleRatio)
   panY += offsetY * (1 - scaleRatio)
   ```

---

## Complete Working Implementation

```javascript
const W = 1920, H = 900;
const svg = d3.select("#viz").attr("width", W).attr("height", H);

// Initialize scale to match scaleExtent minimum
let scale = 3, prevScale = 3, panX = 0, panY = 0;

// Custom projection function
const project = (p) => {
    const s = 180 * scale / (4 + p.z);
    return {
        px: p.x * s + W / 2 + panX,
        py: p.y * s + H / 2 + panY
    };
};

const render = () => { /* ... render using project() ... */ };

// Zoom with cursor-centering
svg.call(d3.zoom()
    .scaleExtent([3, 300])
    .filter(e => e.type === 'wheel' || e.type === 'dblclick')
    .on("zoom", (e) => {
        const newScale = e.transform.k;
        const scaleRatio = newScale / prevScale;
        
        // Get mouse position relative to SVG
        const [mouseX, mouseY] = d3.pointer(e, svg.node());
        
        // Calculate offset from visual center
        const offsetX = mouseX - (W / 2 + panX);
        const offsetY = mouseY - (H / 2 + panY);
        
        // Adjust pan to keep point under cursor fixed
        panX += offsetX - offsetX * scaleRatio;
        panY += offsetY - offsetY * scaleRatio;
        
        // Update scales
        prevScale = newScale;
        scale = newScale;
        render();
    }));
```

---

## Key Takeaways

| Issue | Symptom | Fix |
|-------|---------|-----|
| Scale mismatch | Visualization jumps/disappears on first zoom | Initialize `scale` to match `scaleExtent[0]` |
| Wrong ratio | Zoom drifts from cursor over time | Track `prevScale` separately |
| Wrong pan formula | Content slides away from cursor | Use offset-from-center formula |
| No filter | Scroll hijacks page scrolling | Add `.filter(e => e.type === 'wheel')` |

---

## Bonus: Combining with CTRL+Drag Pan

If you also want CTRL+drag for panning, see [D3JS-Fixing-Control-Handling.md](./D3JS-Fixing-Control-Handling.md) for handling D3's default drag filter that blocks modifier keys.

```javascript
// Global CTRL tracking
let ctrlPressed = false;
document.addEventListener('keydown', e => { if (e.key === 'Control') ctrlPressed = true; });
document.addEventListener('keyup', e => { if (e.key === 'Control') ctrlPressed = false; });

// Drag with CTRL support
svg.call(d3.drag()
    .filter(e => !e.button)  // Allow CTRL+drag
    .on("drag", e => {
        if (ctrlPressed) {
            panX += e.dx;
            panY += e.dy;
        } else {
            // Rotate or other default behavior
        }
        render();
    }));
```

---

## References

- [D3 Zoom Documentation](https://d3js.org/d3-zoom)
- [d3.pointer() Reference](https://d3js.org/d3-selection/pointer)
