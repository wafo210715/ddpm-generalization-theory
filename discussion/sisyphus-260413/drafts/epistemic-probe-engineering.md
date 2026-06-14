# Engineering Guide: Epistemic Probe Framework

> Companion to `.sisyphus/plans/epistemic-probe-framework.md`
> Purpose: Code-level specification for data flow, module structure, testing, experiments, and visualization
> Audience: Developer implementing the plan (human or AI agent)

---

## 0. Critical Codebase Gaps (MUST FIX — validated by explore agents 2026-04-13)

> These discrepancies were found by cross-referencing the engineering guide against actual code.
> Implementers MUST address these in their tasks.

### GAP-1: train.py hardcodes model_channels=64 (line 80)
```python
# CURRENT (train.py:80):
ddpm_unet = DDPMUnet(config.img_channels, 64)  # Hardcoded 64!
# ALSO in evaluation/memorization.py:218:
ddpm_unet = DDPMUnet(config.img_channels, 64)  # Same hardcoded 64!
```
**Impact**: Config field `base_channels` is NOT wired through. Changing YAML won't affect model.
**Fix (Task 5)**: Either read `config.base_channels` or keep 64 and document the decision.
**Recommendation**: Use 32 for pilot (memorization-sensitive), 64 for full training.

### GAP-2: GeoSpatialDataset returns Tensor, not dict
```python
# CURRENT (datasets/dataset.py:173):
def __getitem__(self, idx: int) -> Tensor:
    # Returns plain tensor — no cell_id, no metadata
```
**Impact**: Cannot trace generated samples back to specific urban cells.
**Fix (Task 5)**: Return `{'image': Tensor, 'cell_id': str}` and update train.py's batch handling.

### GAP-3: Cell naming is sequential ({label:04d}.png), not grid-based
```python
# CURRENT (build_hcmc_pipeline.py:820-896):
# Saves as CELLS_256_DIR/{label:04d}.png where label is sequential integer
```
**Impact**: Cannot map images back to spatial grid positions.
**Fix (Task 2)**: V2 pipeline saves as `{row:03d}_{col:03d}.png`.

### GAP-4: Building channel is binary, not density
```python
# CURRENT (build_hcmc_pipeline.py:255-283):
# compute_building_channel() returns binary presence (1.0 if intersects, 0.0 if not)
# or rasterize_buildings() burns to binary (255 or 0)
```
**Impact**: Binary building data loses density variation — critical for urban vitality differentiation.
**Fix (Task 2)**: V2 pipeline computes building footprint coverage per 50m buffer ring (continuous [0,1]).

### GAP-5: HCMC_BBOX undefined
```python
# CURRENT (build_hcmc_pipeline.py:96):
# References HCMC_BBOX but it's NOT DEFINED anywhere in the codebase
```
**Impact**: Pipeline cannot run without this constant.
**Fix (Task 1)**: Define HCMC full municipality bbox OR switch to osmnx geocode approach.

### GAP-6: v1 pipeline normalizes per-canvas-max, NOT per-channel global percentile
```python
# CURRENT (build_hcmc_pipeline.py:820-896):
# G channel: normalize by G.max() (per-canvas)
# R channel: normalize by R.max() (per-canvas)
# B channel: already binary [0, 255]
```
**Impact**: Per-canvas max normalization is NOT the agreed approach. User specified per-channel global 95th percentile.
**Fix (Task 2)**: V2 pipeline computes global percentile across all cells per channel.

### GAP-7: No attention in U-Net by default
```python
# CURRENT (model/unet.py:253):
is_attn: list[bool] = [False, False, False, False]  # NO attention at any level
```
**Impact**: Lower model capacity — may be intentional for memorization research.
**Decision**: Keep attention disabled for pilot. Enable only if model underfits.

---

## 1. Project Directory Structure (Target)

```
ddpm-generalization-theory/
├── data-processing/
│   ├── scripts/
│   │   ├── build_hcmc_pipeline.py      # EXISTING (v1 pipeline — reference only)
│   │   ├── build_hcmc_pipeline_v2.py   # NEW — redesigned pipeline
│   │   ├── pre_encoding_analysis.py    # NEW — PCA + typology
│   │   ├── diagnose_pipeline.py        # EXISTING (debug tool)
│   │   └── plot_poi_distribution.py    # EXISTING (visualization)
│   ├── data/
│   │   └── hcmc/
│   │       ├── raw/
│   │       │   ├── hcmc_boundary.gpkg         # EXISTING (old urban core)
│   │       │   ├── hcmc_boundary_full.gpkg    # NEW (full municipality)
│   │       │   ├── hcmc_grid_1km.gpkg         # NEW (fishnet covering HCMC)
│   │       │   ├── hcmc_grid_500m.gpkg        # NEW (future)
│   │       │   ├── hcmc_grid_200m.gpkg        # NEW (future)
│   │       │   ├── hcmc_buildings.gpkg        # EXISTING/UPDATED
│   │       │   ├── hcmc_pois.gpkg             # EXISTING/UPDATED
│   │       │   └── hcmc_network.graphml       # EXISTING/UPDATED
│   │       ├── processed/
│   │       │   ├── cells/                      # NEW — per-cell RGB PNGs (output of pipeline)
│   │       │   ├── channels/                   # NEW — per-channel B/W PNGs (for visualization)
│   │       │   ├── splits/                     # NEW — train/val/test CSVs
│   │       │   ├── b_canvas.npy               # EXISTING (old)
│   │       │   └── g_canvas.npy, r_canvas.npy # EXISTING (old)
│   │       └── reports/
│   │           ├── pre_encoding/               # NEW — PCA, typology reports
│   │           └── (existing quality reports)
│   └── tests/
│       ├── test_boundary.py              # NEW — boundary/grid validation
│       ├── test_rasterization.py         # NEW — channel computation tests
│       ├── test_normalization.py         # NEW — normalization tests
│       └── conftest.py                   # NEW — shared fixtures (fake data)
│
├── ddpm-win-master/
│   ├── configs/
│   │   ├── default.yaml                  # EXISTING (MNIST)
│   │   ├── geospatial.yaml              # EXISTING (old geo config)
│   │   └── hcmc_vitality_v2.yaml        # NEW — HCMC training config
│   ├── datasets/
│   │   └── dataset.py                    # NEW/UPDATE — GeoSpatialDataset
│   ├── model/
│   │   ├── ddpm.py                       # EXISTING — DDPM core
│   │   ├── unet.py                       # EXISTING — U-Net
│   │   └── diffusion_config.py           # EXISTING — schedule configs
│   ├── evaluation/
│   │   ├── memorization.py              # EXISTING — F_mem (ported)
│   │   ├── compute_fmem_batch.py        # NEW — batch F_mem across checkpoints
│   │   └── fid.py                       # NEW — FID computation (experimental)
│   ├── sampling/
│   │   └── ddim.py                       # EXISTING — DDIM sampling
│   ├── analysis/
│   │   ├── section_2_4/                  # NEW — analysis outputs
│   │   ├── intrinsic_dim.py             # NEW — PCA intrinsic dimension
│   │   └── visualization.py             # NEW — plot specs
│   ├── tests/
│   │   ├── test_all.py                   # EXISTING
│   │   ├── test_config_yaml.py          # EXISTING
│   │   └── test_dataset.py              # NEW — dataset loader tests
│   ├── train.py                          # EXISTING — training loop
│   ├── sample.py                         # EXISTING — sampling
│   └── config_loader.py                 # EXISTING — config system
│
├── .sisyphus/
│   ├── plans/epistemic-probe-framework.md
│   ├── drafts/epistemic-probe-engineering.md  # THIS FILE
│   └── evidence/                              # QA evidence outputs
```

---

## 2. Data Flow End-to-End

```
[OSM API]
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ WAVE 1: DATA COLLECTION & PIPELINE                  │
│                                                     │
│ 1. Boundary + Fishnet                               │
│    ox.geocode_to_gdf("HCMC") → boundary GPKG       │
│    fishnet(1km, covering boundary) → grid GPKG      │
│                                                     │
│ 2. OSM Data Fetch (per fishnet cell)                │
│    Overpass → buildings.gpkg (polygons)             │
│    OSMnx → network.graphml (graph)                  │
│    Overpass → pois.gpkg (points + ways)             │
│                                                     │
│ 3. Channel Computation (per 50m calculation unit)   │
│    For each 1km cell:                               │
│      B: building density per 50m unit (buffer ring) │
│         → 20×20 density map → upsample 128×128      │
│      G: POI Shannon entropy per 50m unit (buffer)   │
│         → 20×20 entropy map → upsample 128×128      │
│      R: accessibility per 50m unit (gravity decay)  │
│         → 20×20 access map → upsample 128×128       │
│                                                     │
│ 4. Normalization (per-channel global percentile)    │
│    B_p95 = percentile_95(B across ALL cells)        │
│    G_p95 = percentile_95(G across ALL cells)        │
│    R_p95 = percentile_95(R across ALL cells)        │
│    B_norm = clip(B / B_p95, 0, 1) × 255            │
│    G_norm = clip(G / G_p95, 0, 1) × 255            │
│    R_norm = clip(R / R_p95, 0, 1) × 255            │
│                                                     │
│ 5. RGB Fusion + Save                                │
│    RGB = stack(B_norm, G_norm, R_norm) → PNG        │
│    Save per-cell PNG → data/hcmc/processed/cells/   │
│    Save per-channel B/W → data/hcmc/processed/channels/ │
│                                                     │
│ 6. Cell Filtering                                   │
│    Exclude cells where all channels are >90% black   │
│    Report: total cells, valid cells, excluded cells  │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ WAVE 2: DATASET PREP + PRE-ENCODING ANALYSIS        │
│                                                     │
│ 7. Pre-encoding Analysis                            │
│    Load all cell PNGs → flatten → PCA               │
│    → scree plot, intrinsic dim, variance explained   │
│    Extract per-cell scalars → k-means clustering     │
│    → typology labels, silhouette scores             │
│                                                     │
│ 8. Dataset + Config                                 │
│    GeoSpatialDataset: load PNGs, train/val/test split│
│    hcmc_vitality_v2.yaml: training configuration    │
│    DataLoader → (B, 3, 128, 128) in [-1, 1]        │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ WAVE 3: PILOT (GATE) + F_mem                        │
│                                                     │
│ 9. Pilot Training (1000 steps × 3 resolutions)     │
│    For res in [64, 128, 256]:                       │
│      Train 1000 steps, checkpoint at [100,250,...]  │
│      For each checkpoint:                           │
│        Sample 50 images                             │
│        Compute F_mem vs training set                │
│        Compute FID vs training set (experimental)   │
│      Plot F_mem curve, FID curve                    │
│    GATE: select best resolution                     │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ WAVE 4: FULL TRAINING + ANALYSIS                    │
│                                                     │
│ 10. Full Training (50K steps, selected resolution)  │
│     Checkpoints at log intervals                    │
│     Generate 50 samples per checkpoint              │
│                                                     │
│ 11. Section 2.4 Analysis                            │
│     F_mem curve → generalization window             │
│     Per-channel statistics per checkpoint            │
│     Joint B/G/R distribution evolution              │
│     Synergy analysis (correlation, MI)              │
│     Threshold comparison with literature             │
│                                                     │
│ 12. Visualization                                   │
│     All plots saved to analysis/ directory          │
└─────────────────────────────────────────────────────┘
```

---

## 3. Module Specifications

### 3.1 `build_hcmc_pipeline_v2.py` — Data Pipeline

#### Function: `get_hcmc_boundary() → gpd.GeoDataFrame`
```
Input: None (queries OSMnx)
Output: GeoDataFrame with full HCMC municipality boundary
Side effects: Saves to hcmc_boundary_full.gpkg (cached)
API: ox.geocode_to_gdf("Ho Chi Minh City, Vietnam")
CRS: EPSG:4326 (WGS84) → converted to EPSG:32648 (UTM 48N) for grid work
```

#### Function: `build_fishnet_grid(boundary_gdf, cell_size_km=1.0) → gpd.GeoDataFrame`
```
Input: boundary GeoDataFrame, cell size in km
Output: GeoDataFrame of rectangular grid cells covering boundary
Logic:
  1. Convert boundary to UTM (EPSG:32648)
  2. Get total_bounds → [xmin, ymin, xmax, ymax]
  3. Snap xmin, ymin DOWN to nearest 1000m multiple (grid alignment)
  4. Snap xmax, ymax UP to nearest 1000m multiple
  5. Generate fishnet: rows = (ymax-ymin)/cell_size, cols = (xmax-xmin)/cell_size
  6. Each cell = shapely.box(x, y, x+cell_size, y+cell_size)
  7. Assign cell_id = f"{row:03d}_{col:03d}"
  8. DO NOT clip to boundary — fishnet extends beyond
Side effects: Saves to hcmc_grid_1km.gpkg
```

#### Function: `collect_buildings_for_grid(grid_gdf) → gpd.GeoDataFrame`
```
Input: fishnet grid GeoDataFrame
Output: building footprint polygons for entire grid extent
Logic:
  1. Get grid total_bounds in WGS84
  2. Split into tiles (avoid Overpass timeout, ~20km per tile)
  3. For each tile: query Overpass for building ways/relations
  4. Merge all tiles → single buildings GeoDataFrame
  5. Add cell_id to each building (which cell it belongs to)
CRITICAL: Fetch data for FISHNET EXTENT, not boundary
```

#### Function: `collect_pois_for_grid(grid_gdf) → gpd.GeoDataFrame`
```
Same pattern as buildings but for POI nodes/ways.
Categories: amenity, shop, office, tourism, leisure (from existing pipeline)
```

#### Function: `collect_network_for_grid(grid_gdf) → nx.MultiDiGraph`
```
Same pattern using OSMnx: ox.graph_from_polygon(grid_extent, network_type="walk")
```

#### Function: `compute_building_channel(buildings, grid_gdf, calc_unit_m=50) → np.ndarray`
```
Input: buildings GPKG, grid GPKG, calculation unit size in meters
Output: per-cell 20×20 density maps (flattened list of arrays)
Logic (for each 1km cell):
  1. Divide cell into 20×20 sub-grid (50m units)
  2. For each 50m unit center:
     a. Create buffer ring (radius TBD, likely 100-200m)
     b. Calculate building footprint coverage within ring
     c. coverage = sum(building.area within ring) / ring.area
  3. Result: 20×20 array of coverage values
Normalization: Applied LATER (see normalize_channels)
```

#### Function: `compute_diversity_channel(pois, grid_gdf, calc_unit_m=50) → np.ndarray`
```
Same pattern but:
  For each 50m unit center:
    a. Create buffer ring
    b. Collect all POIs within ring
    c. Group by category → count per category
    d. Shannon entropy: -sum(p_i * log(p_i)) where p_i = count_i / total
  Result: 20×20 array of entropy values
```

#### Function: `compute_accessibility_channel(network, grid_gdf, calc_unit_m=50) → np.ndarray`
```
Same pattern but:
  1. Build pandana Network from graph
  2. For each 50m unit center:
     a. Snap to nearest network node
     b. Compute gravity-model accessibility: sum(exp(-beta * distance) * weight)
     c. beta = 0.3/km (from existing pipeline)
  Result: 20×20 array of accessibility scores
```

#### Function: `normalize_channels(b_maps, g_maps, r_maps, percentile=95) → tuple`
```
Input: lists of 20×20 arrays per channel, across all cells
Output: normalized arrays per channel per cell
Logic:
  1. Flatten all B values across all cells → B_flat
  2. B_p95 = np.percentile(B_flat, percentile)
  3. B_norm = np.clip(B / B_p95, 0, 1) × 255  (for each cell)
  4. Repeat for G, R
  5. Return (B_norm_list, G_norm_list, R_norm_list)
This balances channels so no single channel dominates the fused image.
```

#### Function: `upsample_and_fuse(b_norm, g_norm, r_norm, target_px=128) → np.ndarray`
```
Input: three 20×20 normalized arrays per cell
Output: 128×128×3 uint8 RGB image
Logic:
  1. B_up = scipy.ndimage.zoom(b_norm, target_px/20, order=1)  # bilinear
  2. G_up = scipy.ndimage.zoom(g_norm, target_px/20, order=1)
  3. R_up = scipy.ndimage.zoom(r_norm, target_px/20, order=1)
  4. RGB = np.stack([R_up, G_up, B_up], axis=2).astype(np.uint8)  # OpenCV convention
  5. Return RGB
NOTE: R=accessibility, G=diversity, B=density (OpenCV BGR convention in file save,
      but PIL saves as RGB. Verify channel order matches config.)
```

### 3.2 `pre_encoding_analysis.py` — Pre-encoding Analysis

#### Function: `run_pca(cell_dir, output_dir)`
```
Input: directory of cell PNGs
Output: scree plot, variance report, intrinsic dimension estimate
Logic:
  1. Load all PNGs → flatten each to 1D (128×128×3 = 49152 dims)
  2. Stack into matrix X (n_cells × 49152)
  3. Fit PCA: sklearn.decomposition.PCA()
  4. Compute cumulative variance → find k where cumvar ≥ 0.95
  5. k = intrinsic dimension estimate
  6. Save: scree plot, variance plot, report
```

#### Function: `classify_typology(cell_dir, output_dir, k_range=range(3,8))
```
Input: directory of cell PNGs
Output: typology labels CSV, silhouette scores
Logic:
  1. For each cell, extract scalar features:
     - B_mean, G_mean, R_mean (average channel values)
     - B_std, G_std, R_std (channel variability)
     - B_moran, G_moran, R_moran (spatial autocorrelation)
  2. Fit k-means for each k in k_range
  3. Compute silhouette score for each k
  4. Select k with highest silhouette
  5. Assign labels → save CSV with cell_id, typology_label
```

### 3.3 `datasets/dataset.py` — GeoSpatialDataset

```python
class GeoSpatialDataset(Dataset):
    """
    Loads per-cell RGB PNGs for DDPM training.
    
    CURRENT STATE (datasets/dataset.py, 234 lines):
      - __getitem__ returns Tensor (C, H, W) directly — NOT a dict
      - Uses torchvision transforms: ToTensor → Resize → Normalize(center, normalize)
      - Normalization: [0,1] → center → [-0.5, 0.5] → normalize → [-1, 1]
      - Split: deterministic by sorted filename order (seed=42)
      - train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
    
    V2 CHANGES NEEDED (Task 5):
      - Return dict with {'image': Tensor, 'cell_id': str} for traceability
      - Accept split_file CSV parameter for explicit train/val/test splits
      - cell_id extracted from filename (currently {label:04d}.png format)
    
    Args:
        root_dir: Directory containing PNG files
        img_size: Target image size (default 256)
        split: 'train', 'val', or 'test'
        train_ratio: Training split ratio (default 0.8)
        val_ratio: Validation split ratio (default 0.1)
        center: Subtract 0.5 (default True)
        normalize: Scale to [-1, 1] (default True)
        seed: Reproducibility (default 42)
    
    Current __getitem__ returns:
        Tensor (3, H, W) float32 in [-1, 1]
    
    V2 __getitem__ should return:
        {'image': Tensor (3, H, W) in [-1, 1], 'cell_id': str}
    """
```

### 3.4 `evaluation/compute_fmem_batch.py` — Batch F_mem

```python
def compute_fmem_across_checkpoints(
    checkpoint_dir: str,      # Directory of .pth checkpoints
    train_dataset: Dataset,    # Training images as tensors
    samples_per_ckpt: int = 50,
    gap_threshold: float = 1/3,
    output_csv: str = None,    # Output CSV path
) -> pd.DataFrame:
    """
    For each checkpoint in checkpoint_dir:
      1. Load model
      2. Generate samples_per_ckpt images
      3. Compute F_mem against train_dataset using memorization.py
      4. Record (step, f_mem, ci_lower, ci_upper)
    
    Returns DataFrame with F_mem at each checkpoint.
    Saves CSV and F_mem curve plot.
    """

def compute_fid_across_checkpoints(
    checkpoint_dir: str,
    train_dataset: Dataset,
    samples_per_ckpt: int = 50,
    output_csv: str = None,
) -> pd.DataFrame:
    """
    EXPERIMENTAL: Compute FID at each checkpoint.
    Uses InceptionV3 features (standard FID).
    Purpose: Validate whether FID is meaningful for urban rasters.
    """
```

### 3.5 `analysis/section_2_4/` — Analysis Modules

```python
# channel_statistics.py
def compute_channel_stats(samples: List[np.ndarray]) -> pd.DataFrame:
    """
    For each sample, dissect into B/G/R channels.
    Compute: mean, std, min, max, Moran's I per channel.
    Returns DataFrame with per-sample stats.
    """

# synergy_analysis.py  
def compute_synergy_metrics(
    sample_stats: pd.DataFrame,  # From channel_statistics
    training_stats: pd.DataFrame,
) -> dict:
    """
    Compute joint distribution metrics:
    - B-G correlation (density-diversity synergy)
    - B-R correlation (density-accessibility synergy)
    - G-R correlation (diversity-accessibility synergy)
    - Mutual information between channel pairs
    - 3D histogram of (B_mean, G_mean, R_mean)
    Returns dict of synergy metrics.
    """

# threshold_comparison.py
def compare_with_literature(
    sample_stats: pd.DataFrame,
) -> dict:
    """
    Compare sampled distributions with empirical thresholds:
    - Jinan: POI blending > 1.5 (suppression → promotion)
    - Beijing: intersection density optimal 22-28/km²
    - General: density inverted-U peak at 0.41-0.497
    
    Map channel values to physical quantities using normalization params.
    Check if model independently reproduces threshold patterns.
    """
```

---

## 4. Testing Strategy

### 4.1 Test Fixtures (`data-processing/tests/conftest.py`)

```python
import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import box, Point, Polygon

@pytest.fixture
def fake_boundary():
    """Small rectangular boundary for testing (100m × 100m in UTM)."""
    return gpd.GeoDataFrame(
        {"name": ["test"]},
        geometry=[box(600000, 1200000, 600100, 1200100)],
        crs="EPSG:32648"
    )

@pytest.fixture
def fake_grid(fake_boundary):
    """1km fishnet covering the fake boundary."""
    # Returns 2×2 grid of 1km cells
    ...

@pytest.fixture
def fake_buildings(fake_grid):
    """Random building polygons within grid cells."""
    # 10-20 buildings per cell, random footprints
    ...

@pytest.fixture
def fake_pois(fake_grid):
    """Random POI points within grid cells."""
    # 5-15 POIs per cell with random categories
    ...

@pytest.fixture
def fake_channel_data():
    """20×20 arrays simulating channel computation output."""
    b_map = np.random.uniform(0, 0.8, (20, 20))
    g_map = np.random.uniform(0, 3.5, (20, 20))
    r_map = np.random.uniform(0, 1.0, (20, 20))
    return b_map, g_map, r_map

@pytest.fixture
def fake_cell_images(tmp_path):
    """Generate 20 fake 128×128×3 PNG images for testing."""
    for i in range(20):
        img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        Image.fromarray(img).save(tmp_path / f"cell_{i:04d}.png")
    return tmp_path
```

### 4.2 Test Files

| Test File | What It Tests | Key Assertions |
|-----------|--------------|----------------|
| `test_boundary.py` | Boundary + fishnet generation | Area ≈ 2000 km², grid aligned to 1000m, cells ≥ 500 |
| `test_rasterization.py` | Channel computation from fake data | 20×20 output shape, values in expected range, Shannon entropy correct |
| `test_normalization.py` | Normalization correctness | Per-channel p95 used, values in [0,255], channels balanced (similar p95-normalized ranges) |
| `test_fusion.py` | RGB stacking + upsample | Output shape 128×128×3, dtype uint8, no channel dominance |
| `test_dataset.py` | DataLoader in ddpm-win-master | Shape (B,3,128,128), values in [-1,1], cell_id preserved |
| `test_fmem.py` | F_mem computation | Known test case: identical images → F_mem=1.0, random images → F_mem≈0.0 |

### 4.3 Validation Commands (Per Task)

Each task has a validation command. These are Python one-liners that check output correctness:

```bash
# After Task 1 (Boundary):
python -c "import geopandas as gpd; g=gpd.read_file('data/hcmc/raw/hcmc_boundary_full.gpkg'); print(f'Area: {g.to_crs(32648).area.iloc[0]/1e6:.0f} km²')"

# After Task 2 (Pipeline):
python -c "from pathlib import Path; from PIL import Image; import numpy as np; cells=list(Path('data/hcmc/processed/cells').glob('*.png')); print(f'{len(cells)} cells'); vals=[np.array(Image.open(c)) for c in cells[:5]]; print(f'Range: [{min(v.min() for v in vals)}, {max(v.max() for v in vals)}]')"

# After Task 5 (Dataset):
python -c "import sys; sys.path.insert(0,'ddpm-win-master'); from datasets.dataset import GeoSpatialDataset; ds=GeoSpatialDataset('data/hcmc/processed/cells',split='train'); print(f'Train: {len(ds)}'); b=ds[0]['image']; print(f'Shape: {b.shape}, Range: [{b.min():.2f}, {b.max():.2f}]')"
```

---

## 5. Comparative Experiments & Ablation

### 5.1 Resolution Comparison (Pilot — Task 6)

```
Experiment: Train at 64×64, 128×128, 256×256 simultaneously
Fixed variables: U-Net base_channels=32, batch=16, lr=0.0001, linear schedule, 1000 steps
Measured: F_mem at each checkpoint, loss curve, FID

Expected outcome:
  64×64: Wider generalization window (less to memorize), but may lose spatial detail
  128×128: Good balance (pilot will confirm)
  256×256: Narrow window or immediate memorization (too many pixels for ~1000 samples)

Decision rule: Select resolution with widest detectable window (F_mem < 0.3 for most checkpoints)
```

### 5.2 Schedule Comparison (Post-pilot, optional)

```
If time permits, compare:
  - Linear schedule (b0=0.0001, bT=0.02) — Bonnaire et al. default
  - VP schedule (beta_d=19.9, beta_min=0.1) — Song et al.
  - VE schedule (sigma_min=0.02, sigma_max=100.0) — Song et al.

Same training steps, same resolution. Compare F_mem dynamics.
Question: Does schedule type affect generalization window width?
```

### 5.3 Augmentation Ablation (If pilot shows immediate memorization)

```
If F_mem > 0.5 at step 500 at ALL resolutions:
  Try controlled augmentation:
  1. Horizontal flip only
  2. Horizontal + vertical flip
  3. Flip + 90° rotation
  
Measure: Does augmentation widen the window?
Risk: Augmentation adds no new information, just transforms existing data.
       Model may learn invariance to transforms rather than urban structure.
```

### 5.4 Checkpoint Analysis (Section 2.4)

```
For each checkpoint in generalization window:
  1. Generate 50 samples
  2. Compute per-channel stats (B/G/R mean, std, Moran's I)
  3. Compute joint distribution (B_mean × G_mean × R_mean scatter)
  4. Track evolution:
     - Early: channels look like noise (F_mem ≈ 0)
     - Window onset: channel structure emerges (low freq patterns)
     - Window middle: stable urban-like patterns (structural learning)
     - Window end: patterns converge to specific training cells (memorization)
  
This temporal analysis is the CORE of the epistemic probe methodology.
```

---

## 6. Visualization Specifications

### 6.1 Plot Style Guide

```
Color palette:
  - B channel (density): Blues colormap
  - G channel (diversity): Greens colormap
  - R channel (accessibility): Reds colormap
  - RGB fusion: Standard RGB display
  - F_mem curve: #1f77b4 (matplotlib default blue)
  - Loss curve: #ff7f0e (matplotlib default orange)
  - Training samples: grayscale or RGB
  - Generated samples: grayscale or RGB

Plot sizes:
  - Single channel heatmap: 4×4 inches, 150 dpi
  - RGB fusion: 5×5 inches, 150 dpi
  - Time series (F_mem, loss): 8×4 inches, 150 dpi
  - Scatter plots: 6×6 inches, 150 dpi
  - Grid composites: 12×12 inches, 100 dpi

Font: Default matplotlib (DejaVu Sans)
```

### 6.2 Required Plots by Task

| Task | Plot | Type | File |
|------|------|------|------|
| 1 | Boundary + fishnet overview | Map | `reports/boundary_overview.png` |
| 2 | Sample cells: B/G/R channels + RGB fusion | Grid (3 rows × 4 cols) | `reports/sample_cells.png` |
| 2 | Per-channel histogram (all cells) | 3 subplots | `reports/channel_histograms.png` |
| 2 | Cross-channel scatter (B×G, B×R, G×R) | 3 subplots | `reports/channel_scatter.png` |
| 3 | PCA scree plot | Line | `reports/pre_encoding/pca_scree.png` |
| 3 | PCA cumulative variance | Line | `reports/pre_encoding/pca_variance.png` |
| 3 | Typology clusters (B_mean vs G_mean, colored) | Scatter | `reports/pre_encoding/typology_clusters.png` |
| 6 | F_mem vs steps (3 resolutions overlaid) | Line | `logs/pilot/fmem_comparison.png` |
| 6 | Loss vs steps (3 resolutions overlaid) | Line | `logs/pilot/loss_comparison.png` |
| 6 | FID vs steps (3 resolutions) | Line | `logs/pilot/fid_comparison.png` |
| 9 | F_mem vs steps (full training) with window annotated | Line | `analysis/section_2_4/fmem_full.png` |
| 9 | B/G/R mean evolution across checkpoints | 3 subplots, line | `analysis/section_2_4/channel_evolution.png` |
| 9 | B/G/R Moran's I evolution | 3 subplots, line | `analysis/section_2_4/spatial_autocorr.png` |
| 10 | Joint B×G scatter colored by checkpoint | Scatter | `analysis/section_2_4/synergy/bg_scatter.png` |
| 10 | Joint B×R scatter colored by checkpoint | Scatter | `analysis/section_2_4/synergy/br_scatter.png` |
| 10 | Joint G×R scatter colored by checkpoint | Scatter | `analysis/section_2_4/synergy/gr_scatter.png` |
| 10 | Synergy strength (correlation) vs step | Line | `analysis/section_2_4/synergy/synergy_evolution.png` |
| 10 | 3D histogram (B, G, R) at window midpoint | 3D bar | `analysis/section_2_4/synergy/joint_distribution.png` |
| 10 | Training vs sampled distribution comparison | Overlay histograms | `analysis/section_2_4/synergy/dist_comparison.png` |

### 6.3 Special Visualization: Partial Dependence Plots (for non-linear threshold analysis)

```python
def plot_channel_threshold(
    sample_stats: pd.DataFrame,
    channel: str,  # 'B', 'G', or 'R'
    training_stats: pd.DataFrame,
    literature_threshold: float = None,
    output_path: str = None,
):
    """
    Shows non-linear relationship between a channel and its co-occurrence
    with the other two channels.
    
    X-axis: channel value (e.g., B_mean = density)
    Y-axis: average of other two channels (e.g., (G_mean + R_mean) / 2)
    
    If literature_threshold provided, draw vertical line.
    
    Interpretation:
    - If the curve shows inverted-U or threshold pattern → model learned non-linearity
    - If curve is flat → channels are independent (no synergy)
    - If curve is monotonically increasing → "more is better" (linear thinking)
    """
```

---

## 7. Configuration Reference

### `hcmc_vitality_v2.yaml`

```yaml
# DDPM Training Configuration for HCMC Urban Vitality
# Companion to: .sisyphus/plans/epistemic-probe-framework.md

# Data
dataset:
  name: "hcmc_vitality"
  root: "../data-processing/data/hcmc/processed/cells"
  image_size: 128           # Pilot will compare 64/128/256
  channels: 3               # RGB fusion
  normalize: true            # [-1, 1] range
  split_file: "../data-processing/data/hcmc/processed/splits/splits.csv"

# Model
model:
  type: "ddpm"
  unet:
    # NOTE: train.py currently HARDCODES model_channels=64 (line 80).
    # Task 5 must wire this through config or keep the hardcoded value.
    # Using 32 for memorization-sensitive pilot (smaller capacity = wider window).
    # Full training may use 64 if pilot shows sufficient window.
    base_channels: 32        # MUST wire into train.py (currently hardcoded to 64)
    channel_mults: [1, 2, 2, 2]  # Matches DDPMUnet default — [1,2,4,8] is too large
    num_res_blocks: 2
    is_attn: [False, False, False, False]  # Default: no attention (less capacity = less memorization risk)
    # NOTE: Engineering guide originally specified attention_resolutions: [16, 8] with num_heads: 4.
    # This was incorrect — DDPMUnet uses is_attn: list[bool] per level, not resolution-based.
    # Attention ENABLED later if pilot shows model underfitting.
    dropout: 0.1

# Diffusion
diffusion:
  schedule_type: "linear"    # Pilot may compare with VP, VE
  b0: 0.0001
  bT: 0.02
  timesteps: 1000

# Training
training:
  n_steps: 50000
  batch_size: 16
  learning_rate: 0.0001
  optimizer: "adam"
  weight_decay: 0.0
  
  # Logarithmic checkpointing (Bonnaire et al. protocol)
  checkpoint_steps: [100, 250, 500, 1000, 2500, 5000, 10000, 20000, 40000, 50000]
  
  # Dual GPU
  use_dual_gpu: true
  
  # Sampling for evaluation
  eval_samples: 50           # Generate 50 samples per checkpoint
  eval_steps: [100, 250, 500, 1000, 2500, 5000, 10000, 20000, 40000, 50000]

# Logging
logging:
  log_dir: "logs/hcmc_vitality"
  checkpoint_dir: "model_ckpts/hcmc_vitality"
  log_interval: 50           # Log loss every 50 steps
```

---

## 8. Data Contracts (Interfaces Between Modules)

```
Contract 1: Pipeline → Cell PNGs
  Format: PNG, 256×256×3, uint8, [0, 255]
  Naming (v1): {label:04d}.png (sequential: 0000.png, 0001.png, ...)
  Naming (v2): {row:03d}_{col:03d}.png (grid-based: 005_012.png) — enables spatial lookup
  Convention: R=accessibility, G=diversity, B=density (in PIL save order)
  
  NOTE: v1 used sequential labels. v2 should use row_col naming for traceability.
  GeoSpatialDataset v2 must parse filenames to extract cell_id for metadata tracking.
  
Contract 2: Cell PNGs → Dataset Loader
  Format: Tensor, (3, target_size, target_size), float32, [-1, 1]
  Transform: PIL loads [0,255] → ToTensor [0,1] → Resize → Normalize(center=True, normalize=True)
  NOTE: v1 pipeline outputs 256×256 PNGs. Pilot will test 64/128/256 via Resize in DataLoader.
  The RAW data is 256×256; resolution selection happens at DataLoader level.
  
Contract 3: Dataset → Training
  Format: DataLoader yielding Tensor (B, 3, target_size, target_size)
  CURRENT: GeoSpatialDataset.__getitem__ returns plain Tensor — training loop uses directly
  V2: Should return dict{'image': Tensor, 'cell_id': str} — train.py must handle dict
  NOTE: train.py (line 119-126) already handles both tuple and single tensor returns.
       V2 must add dict handling (extract ['image'] from dict).
  
Contract 4: Training → Checkpoints
  Format: .pth file containing:
    - 'eps_model': U-Net state dict (or 'eps_model.module' for DataParallel)
    - 'step': int
    - 'config': dict (training config used)
  
Contract 5: Checkpoints → Samples
  Format: Tensor, (N, 3, target_size, target_size), float32, [-1, 1]
  Transform for visualization: ((tensor + 1) / 2).clamp(0, 1) → (x * 255).to(uint8)
  NOTE: Dataset normalizes to [-1, 1], so samples are also [-1, 1].
       Visualization: ((tensor + 1) * 0.5 * 255) or equivalently ((tensor + 1) / 2 * 255)
  
Contract 6: Samples → Analysis
  Format: numpy arrays after channel dissection
  B channel = samples[:, 0, :, :]  (density)
  G channel = samples[:, 1, :, :]  (diversity)
  R channel = samples[:, 2, :, :]  (accessibility)
```

---

## 9. Error Recovery Guide

### Pipeline breaks at data collection
```
Symptom: Overpass timeout, empty buildings/POIs
Check: data/hcmc/raw/*.gpkg file sizes
Fix: Re-run with smaller tile splits, increase timeout
```

### Pipeline breaks at normalization
```
Symptom: All channels look identical, or one channel is pure white/black
Check: Per-channel percentile values in report
Fix: Adjust percentile (use 90th instead of 95th if outliers dominate)
```

### Training doesn't converge
```
Symptom: Loss stays flat or increases
Check: Learning rate, batch size, data quality
Fix: Reduce lr to 0.00005, check data isn't all zeros
```

### F_mem stays at 0 or jumps to 1 immediately
```
Symptom: No detectable generalization window
Check: Dataset size vs model capacity
Fix: Try smaller resolution (64×64), smaller U-Net (base_channels=16), or augmentation
```

### FID produces NaN or negative values
```
Symptom: FID computation fails
Check: Sufficient samples (need ≥50 per checkpoint)
Finding: If FID consistently fails → document as "FID inapplicable to urban rasters"
```
