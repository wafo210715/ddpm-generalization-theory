"""
Comprehensive benchmark: VRAM + throughput (it/s) for ch=128 across multiple batch sizes.

Adapts batch sizes to GPU VRAM:
  - >= 80 GB (Pro 6000 Blackwell): tests up to bs=256
  - >= 40 GB: tests up to bs=128
  - < 40 GB: tests up to bs=96 (dual 3090 range)

Reports peak VRAM per GPU, iterations/second, and estimated 500K training time.

Usage:
    python benchmark_vram_throughput.py
"""

import torch
from torch.nn.parallel import DataParallel
from model.unet import DDPMUnet
from model.ddpm import DDPM
from model.diffusion_config import DiffusionConfig
import time
import gc


def benchmark(model_channels: int = 128, img_size: int = 128):
    n_gpus = torch.cuda.device_count()
    if n_gpus < 1:
        print("[ERROR] No CUDA GPUs detected.")
        return

    use_dual = n_gpus >= 2

    # Adapt batch size range to GPU VRAM
    gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_vram_gb >= 80:
        batch_sizes = [8, 16, 32, 64, 96, 128, 160]
    elif gpu_vram_gb >= 40:
        batch_sizes = [8, 16, 24, 32, 48, 64, 80, 96, 128]
    else:
        batch_sizes = [8, 16, 24, 32, 48, 64, 80, 96]

    print(f"{'='*80}")
    print(f"BENCHMARK: ch={model_channels}, {img_size}x{img_size}x3, {n_gpus} GPU(s)")
    print(f"GPU: {torch.cuda.get_device_name(0)}, {gpu_vram_gb:.0f} GB VRAM")
    print(f"{'='*80}")
    print(f"{'BS':>4} | {'Per GPU':>8} | {'Peak VRAM':>10} | {'Reserved':>10} | {'it/s':>8} | {'500K time':>12} | {'Status'}")
    print(f"{'-'*4}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*12}-+-{'-'*8}")

    results = []

    for bs in batch_sizes:
        # Cleanup
        gc.collect()
        for i in range(n_gpus):
            with torch.cuda.device(i):
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()

        device = torch.device("cuda:0")

        try:
            # Create model
            ddpm_unet = DDPMUnet(img_channels=3, model_channels=model_channels)
            diffusion_config = DiffusionConfig(
                n_steps=1000, schedule_type="linear", device=str(device),
                b0=0.0001, bT=0.02,
            )
            ddpm = DDPM(1000, ddpm_unet, diffusion_config=diffusion_config)

            if use_dual:
                ddpm.eps_model = DataParallel(ddpm.eps_model, device_ids=list(range(n_gpus)))
            ddpm = ddpm.to(device)

            # Warmup (2 iters)
            ddpm.train()
            for _ in range(2):
                x_0 = torch.randn(bs, 3, img_size, img_size, device=device)
                t = torch.randint(0, 1000, (bs,), device=device)
                eps = torch.randn_like(x_0)
                x_t = ddpm.q_sample(x_0, t, eps)
                eps_theta = ddpm.eps_model(x_t, t)
                loss = torch.nn.functional.mse_loss(eps_theta, eps)
                loss.backward()

            # Reset peak after warmup
            for i in range(n_gpus):
                with torch.cuda.device(i):
                    torch.cuda.reset_peak_memory_stats()

            # Timed benchmark (20 iterations)
            optimizer = torch.optim.AdamW(ddpm.parameters(), lr=0.0001)
            n_bench = 20

            torch.cuda.synchronize()
            t_start = time.perf_counter()

            for _ in range(n_bench):
                x_0 = torch.randn(bs, 3, img_size, img_size, device=device)
                t = torch.randint(0, 1000, (bs,), device=device)
                eps = torch.randn_like(x_0)

                x_t = ddpm.q_sample(x_0, t, eps)
                eps_theta = ddpm.eps_model(x_t, t)
                loss = torch.nn.functional.mse_loss(eps_theta, eps)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ddpm.parameters(), max_norm=1.0)
                optimizer.step()

            torch.cuda.synchronize()
            t_end = time.perf_counter()

            elapsed = t_end - t_start
            its = n_bench / elapsed
            peak_vram = max(
                torch.cuda.max_memory_allocated(i) / 1024**3 for i in range(n_gpus)
            )
            reserved = max(
                torch.cuda.memory_reserved(i) / 1024**3 for i in range(n_gpus)
            )

            # Estimate 500K training time
            total_seconds = 500_000 / its
            hours = total_seconds / 3600

            if peak_vram >= gpu_vram_gb:
                status = "OOM"
            elif peak_vram > gpu_vram_gb * 0.9:
                status = "TIGHT"
            else:
                status = "OK"

            per_gpu = bs // n_gpus if use_dual else bs
            print(f"{bs:>4} | {per_gpu:>7} | {peak_vram:>8.2f} GB | {reserved:>8.2f} GB | {its:>7.2f} | {hours:>9.1f} h | {status}")

            results.append((bs, per_gpu, peak_vram, reserved, its, hours, status))

            # Cleanup
            del ddpm, optimizer, x_0, t, eps, x_t, eps_theta, loss
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            per_gpu = bs // n_gpus if use_dual else bs
            print(f"{bs:>4} | {per_gpu:>7} | {'OOM':>10} | {'---':>10} | {'---':>8} | {'---':>12} | OOM")
            results.append((bs, per_gpu, 99, 99, 0, 0, "OOM"))
            gc.collect()
            torch.cuda.empty_cache()
            continue

    print(f"{'='*80}")
    print()
    print("RECOMMENDATION:")
    # Find highest bs that's OK (not TIGHT or OOM)
    safe = [(bs, its, hrs, vram) for bs, _, vram, _, its, hrs, st in results if st == "OK"]
    if safe:
        best_bs, best_its, best_hrs, best_vram = max(safe, key=lambda x: x[1])
        print(f"  Best throughput: bs={best_bs} ({best_its:.1f} it/s, {best_vram:.1f} GB peak, ~{best_hrs:.0f}h for 500K)")
    # Also show the TIGHT option if available
    tight = [(bs, its, hrs, vram) for bs, _, vram, _, its, hrs, st in results if st == "TIGHT"]
    if tight:
        t_bs, t_its, t_hrs, t_vram = max(tight, key=lambda x: x[1])
        print(f"  Aggressive:      bs={t_bs} ({t_its:.1f} it/s, {t_vram:.1f} GB peak, ~{t_hrs:.0f}h for 500K)")
    print()


if __name__ == "__main__":
    benchmark()
