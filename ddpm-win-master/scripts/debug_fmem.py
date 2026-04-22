"""Debug script: sanity check + gap ratio distribution for F_mem investigation."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from evaluation.memorization import compute_memorization_fraction
from datasets.dataset import get_dataset
from torch.utils.data import DataLoader
from model.unet import DDPMUnet
from model.ddpm import DDPM
from model.diffusion_config import DiffusionConfig
from sampling.ddim import sample_ddim
from config_loader import TrainingConfig


def main():
    config = TrainingConfig.from_yaml('configs/hcmc_vitality_v3_5090.yaml')
    diff_cfg = DiffusionConfig(n_steps=1000, schedule_type='linear', b0=0.0001, bT=0.02, device='cpu')

    # Load training data
    train_ds = get_dataset('hcmc_vitality_v2', img_size=128,
                           data_dir='../data-processing/data/hcmc/processed/cells',
                           split='train')
    loader = DataLoader(train_ds, batch_size=len(train_ds), shuffle=False)
    batch = next(iter(loader))
    train_imgs = batch['image'] if isinstance(batch, dict) else batch
    print(f"Training: {train_imgs.shape}")

    # Load model and generate
    unet = DDPMUnet(3, 128)
    ddpm = DDPM(1000, unet, diffusion_config=diff_cfg)
    ckpt = torch.load(
        'model_ckpts/hcmc_vitality_v3/hcmc_vitality_v2_128x128_steps500000_step500000.pth',
        map_location='cpu', weights_only=False
    )
    if 'eps_model' in ckpt:
        ddpm.eps_model.load_state_dict(ckpt['eps_model'])

    torch.manual_seed(42)
    result = sample_ddim(ddpm, n_images=50, config=config, diffusion_config=diff_cfg, device='cpu')
    gen_imgs = result[0]
    print(f"Generated: {gen_imgs.shape}")

    # ================================================================
    # SANITY CHECK: Inject 5 training images into generated set
    # ================================================================
    print("\n" + "=" * 60)
    print("SANITY CHECK: Inject 5 training images into generated set")
    print("=" * 60)

    injected = gen_imgs.clone()
    inject_indices = [0, 10, 20, 30, 40]
    for i, idx in enumerate(inject_indices):
        injected[idx] = train_imgs[i * 100]

    sanity = compute_memorization_fraction(
        train_imgs, injected, gap_threshold=1.0/3.0, device='cpu'
    )
    fm = sanity['fraction_mem']
    nm = sanity['n_memorized']
    nt = sanity['n_total']
    print(f"With 5 injected: F_mem={fm:.4f}, n_memorized={nm}/{nt}")

    for idx in inject_indices:
        gr = sanity['gap_ratios'][idx]
        is_det = "DETECTED" if gr < 1.0/3.0 else "MISSED"
        print(f"  idx={idx}: gap_ratio={gr:.6f}  [{is_det}]")

    # ================================================================
    # GAP RATIO DISTRIBUTION (clean 50 samples)
    # ================================================================
    print("\n" + "=" * 60)
    print("GAP RATIO DISTRIBUTION (clean 50 samples)")
    print("=" * 60)

    clean = compute_memorization_fraction(
        train_imgs, gen_imgs, gap_threshold=1.0/3.0, device='cpu'
    )
    ratios = clean['gap_ratios']
    fm_clean = clean['fraction_mem']
    nm_clean = clean['n_memorized']
    print(f"F_mem={fm_clean:.4f}, n_memorized={nm_clean}/50")
    print(f"Min={ratios.min():.4f}, Max={ratios.max():.4f}")
    print(f"Mean={ratios.mean():.4f}, Median={np.median(ratios):.4f}")
    pcts = np.percentile(ratios, [10, 25, 50, 75, 90])
    print(f"Percentiles: p10={pcts[0]:.4f}, p25={pcts[1]:.4f}, p50={pcts[2]:.4f}, p75={pcts[3]:.4f}, p90={pcts[4]:.4f}")

    # Histogram
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 5.0]
    hist, _ = np.histogram(ratios, bins=bins)
    print("\nHistogram:")
    for i in range(len(hist)):
        bar = "#" * hist[i]
        print(f"  [{bins[i]:.1f}, {bins[i+1]:.1f}): {hist[i]:3d}  {bar}")

    # Print all 50 gap ratios sorted
    print("\nAll 50 gap ratios (sorted):")
    sorted_r = np.sort(ratios)
    for i, r in enumerate(sorted_r):
        marker = " <--- < 1/3" if r < 1.0/3.0 else ""
        print(f"  {i+1:2d}. {r:.4f}{marker}")


if __name__ == '__main__':
    main()
