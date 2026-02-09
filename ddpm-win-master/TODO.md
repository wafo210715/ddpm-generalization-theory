# Integration Progress - DDPM Research Features

## Workflow: Code → Validate → Document → Repeat

---

## SYSTEM CONFIGURATION (DISCOVERED)

**Platform**: Windows 10 (AMD64)
**Python**: 3.10.11
**Environment**: uv virtual environment (.venv/)
**PyTorch**: 2.7.1+cu118 with CUDA 11.8 (installed via uv)

**Environment Management**:
```bash
# Create venv with uv
uv venv

# Activate (Windows)
.venv\Scripts\activate

# Install from lock file (reproducible)
uv pip sync uv.lock

# Add new package
uv pip add <package>

# Update lock file
uv pip compile requirements.txt -o uv.lock
```

**Files Created**:
- `uv.lock` - Pinned dependencies for reproducible builds
- `requirements.txt` - High-level dependency spec
- `ENVIRONMENT.md` - Setup documentation

**GPU Setup** (EXCELLENT for multi-GPU training!):
```
GPU 0: NVIDIA GeForce RTX 3090 (24GB)
  - 3x NVLink @ 14.062 GB/s = ~42 GB/s
  - PCIe Gen 4 x16 @ ~32 GB/s
  - TOTAL: ~74 GB/s interconnect bandwidth
  - UUID: GPU-292787e3-0741-eb21-a1f8-e0f096d5e8e6

GPU 1: NVIDIA GeForce RTX 3090 (24GB)
  - 3x NVLink @ 14.062 GB/s = ~42 GB/s
  - PCIe Gen 4 x16 @ ~32 GB/s
  - TOTAL: ~74 GB/s interconnect bandwidth
  - UUID: GPU-bee32acd-e4c5-f15f-dfbb-83d5579747f5
```

**Multi-GPU Strategy** (user has successfully used before):
```python
from torch.distributed import DistributedDataParallel as DDP
from torch.nn.parallel import DistributedDataParallel as DDP

# Option A (DDP - recommended for NVLink):
strategy = DDPStrategy(process_group_backend='gloo', find_unused_parameters=True)

# Option B (DataParallel - simpler but slower):
model = torch.nn.DataParallel(model)
```

**NVLink Validation Commands** (Windows):
```bash
# Check NVLink status
nvidia-smi nvlink --status

# Run full system check
python check_gpu.py
```

---

## Phase 1: Foundation Setup

### ✓ COMPLETED:
- [x] **System Check**: Windows 10, Python 3.10.11, PyTorch 2.0.1+cu118
- [x] **GPU Detection**: 2x RTX 3090 (24GB each)
- [x] **NVLink Validation**: 3x NVLink @ 14.062 GB/s per GPU (~74 GB/s total per GPU!)
- [x] **Dependencies**: All required packages installed
- [x] **Dataset Loader**: MNIST loads correctly (60,000 samples)
- [x] **Created**: datasets/dataset.py, datasets/__init__.py, model_ckpts/
- [x] **Created**: config_loader.py, test_config.py, check_gpu.py, requirements.txt

### TODO (Next):
- [x] **1.3 Quick test train.py** ✓ PASSED (10 steps, loss: 1.16 → 0.21)
- [x] **1.4 Test sample.py** ✓ PASSED (50 steps training + 1000 denoising, avg loss: 0.186)
- [x] **1.5 Confirm dual GPU training works** ✓ PASSED (20 steps, loss: 1.32 → 0.15, DataParallel)

---

## Phase 2.1: Configuration Classes (AFTER FOUNDATION TESTS)

### ✓ COMPLETED:
- [x] 2.1.1 Test `config_loader.py` (already created) ✓
- [x] 2.1.2 Add YAML loading capability ✓ (from_yaml, to_yaml methods)
- [x] 2.1.3 Create `configs/` directory and YAML templates ✓ (default.yaml, geospatial.yaml)
- [x] 2.1.4 Create `diffusion_config.py` - DiffusionConfig with schedules ✓ (linear, cosine, fixed)
- [x] 2.1.5 Test config loading with YAML ✓ ALL TESTS PASSED
- [x] 2.1.6 Update `train.py` to use config ✓ (with argparse --config support)
- [x] 2.1.7 Add dual GPU support (bool in config) ✓ use_dual_gpu parameter
- [x] 2.1.8 Organize test files in `tests/` subfolder ✓
- [x] 2.1.9 Final test suite validation ✓ ALL TESTS PASSED

---

## Phase 2.2: Beta Schedules

### ✓ COMPLETED:
- [x] 2.2.1 Add VP/VE schedule variants to diffusion_config.py ✓ (discretized for discrete timesteps)
- [x] 2.2.2 Update DDPM class to use DiffusionConfig ✓ (accepts optional diffusion_config parameter)
- [x] 2.2.3 Update config_loader.py with unified parameter names ✓ (matches research repos)
- [x] 2.2.4 Update YAML configs with schedule parameters ✓ (default.yaml, geospatial.yaml)
- [x] 2.2.5 Create tests/test_schedules.py ✓ ALL 5 SCHEDULES PASSED
- [x] 2.2.6 Fix sample.py to handle both checkpoint types ✓ (U-Net only vs full DDPM)
- [x] 2.2.7 Unify parameter names across all files ✓ (matches research repo conventions)
- [x] 2.2.8 Document naming convention rule in CLAUDE.md ✓

### Phase 2.2 Summary:
**Implemented Schedules** (all discretized for discrete timesteps):
- Linear: `b0`, `bT` (DDPM standard)
- Fixed: `beta` (constant)
- VP (Variance Preserving): `beta_d`, `beta_min` (Song et al. SDE)
- VE (Variance Exploding): `sigma_min`, `sigma_max` (Song et al. SDE)

**Note**: Cosine schedule was initially implemented but later removed due to numerical instability issues (2025-02-09). It is not used by either of our primary research repos.

**Key Design Decisions**:
- Discrete timesteps maintained (not continuous time) - better for "generalization window" analysis
- Parameter names match research repos (no prefixes like `vp_beta_d`)
- VP/VE schedules discretized from continuous formulas using sigma → alpha_bar → beta conversion

**Files Modified**:
- `model/diffusion_config.py`: Added VP/VE schedules
- `model/ddpm.py`: Accept optional DiffusionConfig parameter
- `config_loader.py`: Added schedule parameters (unified naming)
- `train.py`: Create and pass DiffusionConfig to DDPM
- `configs/default.yaml`: Added all schedule parameters
- `configs/geospatial.yaml`: Added all schedule parameters
- `sample.py`: Handle both U-Net only and full DDPM checkpoints
- `CLAUDE.md`: Added naming convention rule
- `tests/test_schedules.py`: Comprehensive schedule validation

**Verification**:
- All 5 schedule types pass validation tests
- DDPM initializes correctly with each schedule
- Schedules are deterministic (reproducible)
- Beta values in valid range [0, 1]
- Alpha_cumulative monotonically decreasing

### SKIPPED (Intentionally):
- [ ] 2.2.2 EDM (Karras) schedule variant - SKIPPED (native continuous-time, adds complexity for discrete framework)

---

## Phase 2.3: Schedule Comparison Experiments

### COMPLETED (Working Schedules: Linear, VP, VE)

**Status**: Linear, VP, and VE schedules trained successfully with 10K steps each. Parallel training completed in ~20 minutes using optimal GPU utilization.

### ✓ COMPLETED:
- [x] 2.3.1 Experiment A: Linear vs VP (GPU 0: Linear, GPU 1: VP)
- [x] 2.3.2 Experiment B: VP vs VE (GPU 0: VP, GPU 1: VE)
- [x] 2.3.3 Visual comparison of generated samples
- [x] 2.3.4 Beta schedule comparison plot

### Notes:
**Cosine Schedule (Removed)**: The cosine schedule implementation was initially added but exhibited numerical instability (beta values approaching 1.0 at late timesteps, causing black images during sampling). Since neither of our primary research repos (Why-Diffusion-Models-Don-t-Memorize, Diffusion-Model-Generalizability) use cosine schedule, it was removed from the codebase on 2025-02-09 for cleanliness. Our comparison focuses on Linear, VP, and VE schedules.

### Available Checkpoints (Working):
- Linear:  model_ckpts/schedule_comparison/linear/mnist_linear_steps10000.pth
- VP:      model_ckpts/schedule_comparison/vp/mnist_vp_steps10000.pth
- VE:      model_ckpts/schedule_comparison/ve/mnist_ve_steps10000.pth

### Generated Visualizations:
- model_ckpts/schedule_comparison/visualizations/beta_schedule_comparison.png
- model_ckpts/schedule_comparison/visualizations/Linear_vs_VP_comparison.png
- model_ckpts/schedule_comparison/visualizations/VP_vs_VE_comparison.png

### Comparison Commands:
```bash
# Compare any two working schedules:
python compare_schedules.py --checkpoint1 model_ckpts/schedule_comparison/linear/mnist_linear_steps10000.pth \
                            --checkpoint2 model_ckpts/schedule_comparison/vp/mnist_vp_steps10000.pth \
                            --name1 "Linear" --name2 "VP"

# Plot beta schedule comparison:
python compare_schedules.py --plot-schedules
```


### Outputs:
- Checkpoints: `model_ckpts/schedule_comparison/{schedule_type}/mnist_{schedule_type}_steps10000.pth`
- Logs: `logs/schedule_comparison/{schedule_type}/`

### Visualization Plan:
1. **Visual Sample Comparison** - Generate 9 images from each schedule, side-by-side
2. **Loss Curve Plotting** - Compare training loss progression
3. **Beta Schedule Visualization** - Plot noise level progression for each schedule
4. **Quantitative Metrics** - FID, memorization fraction (when implemented)

---

## Phase 2.4: Checkpointing System

### TODO:
- [ ] 2.4.1 Implement logarithmic checkpoint saving during training
- [ ] 2.4.2 Add checkpoint loading/resume capability
- [ ] 2.4.3 Add checkpoint metadata (step, loss, config)
- [ ] 2.4.4 Test checkpoint save/load cycle

## Phase 2.5: Training Loop Integration
(TBD - after Phase 2.3 complete)

---

## Phase 2.6: Model Format Conversion (HuggingFace/WebUI Integration)

### ✓ COMPLETED:
- [x] 2.6.1 Create `scripts/` directory for utilities ✓
- [x] 2.6.2 Create `convert_checkpoint.py` - bidirectional .pth ↔ .safetensors converter ✓
- [x] 2.6.3 Add `safetensors>=0.4.0` to requirements.txt ✓
- [x] 2.6.4 Create `scripts/__init__.py` for package structure ✓
- [x] 2.6.5 Support metadata preservation during conversion ✓
- [x] 2.6.6 Add batch directory conversion support ✓
- [x] 2.6.7 Add file size reporting for conversions ✓

**Features Implemented**:
- Auto-detect format (.pth or .safetensors)
- Handle different checkpoint structures (state_dict, model key, wrapped formats)
- Preserve non-tensor metadata as JSON
- Single file and batch directory conversion
- Size comparison reporting

**Usage**:
```bash
# Convert single file
python scripts/convert_checkpoint.py --input model_ckpts/model.pth

# Convert directory
python scripts/convert_checkpoint.py --input model_ckpts/ --convert-dir
```

---

## Notes
- **Platform**: Windows 10 (important for path handling)
- **Horsepower**: 64GB total VRAM + ~74 GB/s interconnect per GPU (NVLink + PCIe)
- **DDP Strategy**: User has successfully used `DDPStrategy(process_group_backend='gloo')` before
- **Status**: Phase 2.1 COMPLETE! Phase 2.6 (Model Conversion) COMPLETE!
- **Last Action**: Created checkpoint conversion utility (.pth ↔ .safetensors) for HuggingFace/WebUI integration
- **Next**: Phase 2.2 - Beta Schedules (VP/VE/EDM variants) or Phase 2.3 - Checkpointing System

## Phase 1 Summary:
- All foundation tests passed
- Training: 10-step test (loss: 1.16 → 0.21)
- Sampling: 50-step train + 1000 denoising (avg loss: 0.186)
- Dual GPU: 20-step test with DataParallel (loss: 1.32 → 0.15)

## Phase 2.1 Summary:
- Config system: TrainingConfig with YAML loading
- DiffusionConfig: Multiple schedule types (linear, fixed)
- Updated train.py: Config-based with argparse support
- Dual GPU support: use_dual_gpu bool in config
- Test organization: All tests in tests/ subfolder
- Final validation: ALL TESTS PASSED
