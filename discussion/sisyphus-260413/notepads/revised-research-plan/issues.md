## 2026-04-11 Session 1: Pipeline Rework Attempts

### Status
- Step 1.0 (Pre-flight): COMPLETE - memory 4.3GB, U-Net no changes needed
- Step 1.1 (Pipeline Rework): RESOLVED - canvases were correct, cell image generation had R channel normalization bug

### Bug Analysis (RESOLVED)
The initial diagnosis was WRONG. The canvases (b_canvas.npy, g_canvas.npy, r_canvas.npy) were NOT empty:
- B canvas: 770,392 non-zero pixels (buildings exist)
- G canvas: 1,026,107 non-zero pixels
- R canvas: 42,600,919 non-zero pixels

The real bug: R channel cell images were saved with raw float→uint8 cast (values 0-30) instead of per-cell normalization (0-255). The cell PNGs were produced by an older version of the code before the normalization was added. B and G were correct.

### Fix
Created `data-processing/scripts/fix_cell_images.py` — standalone script that re-slices canvases with proper per-cell normalization. Result:
- 713 cells, 0 all-black
- R: 100% cells have data, all with max=255 (per-cell normalized)
- G: 238/713 (33.4%) have data, all with max=255
- B: 235/713 (33.0%) have buildings, binary 0/255

### Data Quality Report
- Buildings cluster in ~1/3 of city (urban core), density 0-29.5% per cell
- POI entropy mirrors building distribution (33.4% cells)
- Accessibility covers all cells (100%), mean 91% coverage
- Cross-channel correlations weak: B-G=0.21, B-R=-0.20, G-R=-0.26
- All 713 cells have at least one channel with data

### Key Path Notes
- Data lives at `data-processing/data/hcmc/` (NOT `data/hcmc/`)
- ROOT in pipeline = `data-processing/` (Path(__file__).parent.parent)
- Grid: 23 rows × 31 cols, canvas: 5888×7936px, 3.9m/px
- B canvas has `np.flipud()` applied, R and G do not — but alignment is correct after flip

## 2026-04-11 Session 2: Parallel Execution

### Parallelization Strategy
- Step 1.5 (eval scripts) and Step 1.2 (dataset/config) can run in parallel
- Both are independent — eval scripts are pure code ports, dataset loader just needs PNG path
- Step 1.3 (training) depends on 1.2
- Step 1.4 (analysis) depends on 1.3 (needs trained model checkpoints)
