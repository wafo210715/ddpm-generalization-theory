
## 2026-04-11: DDIM Sampling + k-NN Memorization Port

### Key Interface Findings

**DiffusionConfig (model/diffusion_config.py)** stores schedule tensors as plain attributes (NOT registered buffers):
- `sqrt_alpha_cumulative` — shape (T,), `torch.sqrt(alpha_cumulative)`
- `sqrt_one_minus_alpha_cumulative` — shape (T,), `torch.sqrt(1 - alpha_cumulative)`
- `alpha_cumulative` — shape (T,), `torch.cumprod(alpha, dim=0)`
- Device is set via `.to(device)` during `initialize()` but derived tensors inherit the device from `beta`

**DDPM model (model/ddpm.py)** registers `beta`, `alpha`, `alpha_bar` as buffers. At inference time:
- `model.eps_model(x, t)` — raw noise prediction, t is shape (B,)
- `model.alpha_bar[t]` — schedule values, needs `[:, None, None, None]` reshape for 4D broadcast
- Full state dict has keys `beta`, `alpha`, `alpha_bar`; U-Net-only state dict has only eps_model keys

**Research repo `get()` helper** gathers a 1D tensor at timestep indices and reshapes: `(B,) → (B,1,1,1)` for dim=4. Replaced with `_gather_4d()` in sampling/ddim.py.

**Checkpoint loading pattern** (from sample.py):
```python
checkpoint = torch.load(path, map_location=device)
if "eps_model" in checkpoint or "beta" in checkpoint:
    ddpm.load_state_dict(checkpoint)  # full DDPM
else:
    ddpm.eps_model.load_state_dict(checkpoint)  # U-Net only (dual GPU training)
```

### Port Decisions

1. **DDIM sigma_t computation** — kept exactly as in source (uses sqrt_alpha_cum squared ratios). Verified mathematically against Eq. 12 in arXiv:2010.02502.
2. **k-NN distances** — switched from research repo's broadcast `s - X` to `torch.cdist` for better numerical stability and clarity.
3. **evaluate_checkpoint_memorization** — uses lazy imports to avoid circular deps at module level. Accepts Tensor, Dataset, or DataLoader for train data.
4. **Batched k-NN** — processes generated images in chunks (default 100) to limit peak GPU memory.

## Geospatial Dataset Loader (2026-04-11)

- **Created**: `ddpm-win-master/datasets/__init__.py`, `ddpm-win-master/datasets/dataset.py`, `ddpm-win-master/configs/hcmc_vitality.yaml`
- `train.py` calls `get_dataset(config.dataset_name, img_size=config.img_size)` — only passes `name` and `img_size`. No `center`/`normalize`/`data_dir` kwargs are passed from train.py, so dataset must handle normalization internally.
- The `TrainingConfig` has `center` and `normalize` fields but they are NOT wired into `train.py`'s dataset loading. The dataset self-applies [-1,1] normalization.
- `config_loader.py` opens YAML with `open(path, 'r')` without specifying encoding — fails on Windows with non-ASCII chars (em-dash). Use ASCII-only comments in YAML configs.
- HCMC data: 713 PNGs at `data-processing/data/hcmc/processed/cells_256x256/`, named `0000.png`-`0712.png`. Split: 570 train / 71 val / 72 test.
- `torchvision.transforms.Normalize([0.5,...],[0.5,...])` maps [0,1] -> [-1,1] (formula: `(x - mean) / std`).
- Pyright can't infer return type of `torchvision.transforms.Compose.__call__` — use `# type: ignore[return-value]` annotation.
