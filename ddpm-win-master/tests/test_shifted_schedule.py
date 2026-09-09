"""Tests for the resolution-shifted noise schedule (B2 / D1).

    At 1024 px the noise schedule must be shifted and the U-Net needs more
    levels.  Simple Diffusion (Hoogeboom et al. 2023) trained 1024 px in
    pixel space with exactly these two changes.

The property under test is what the shift is for: at a fixed timestep, a larger
image must end up genuinely noisier, and the high-noise end of the trajectory
must actually destroy the signal.

Run: pytest tests/test_shifted_schedule.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from config_loader import TrainingConfig
from model.ddpm import DDPM
from model.diffusion_config import DiffusionConfig

SCHEDULES = ["linear", "vp", "ve", "fixed"]


def alpha_bar(img_size=None, shift=False, schedule="linear", n_steps=1000, ref=64):
    return DiffusionConfig(
        n_steps=n_steps,
        schedule_type=schedule,
        shift_schedule=shift,
        img_size=img_size,
        ref_img_size=ref,
    ).alpha_cumulative


# ---------------------------------------------------------------------------
# The shift itself
# ---------------------------------------------------------------------------


def test_shifting_to_the_reference_resolution_is_the_identity():
    base = alpha_bar()
    shifted = alpha_bar(img_size=64, shift=True, ref=64)
    assert torch.allclose(base, shifted, atol=1e-6)


@pytest.mark.parametrize("img_size", [128, 256, 512, 1024])
def test_a_larger_image_is_noisier_at_every_timestep(img_size):
    """The reason the shift exists: the standard schedule under-noises."""
    base = alpha_bar()
    shifted = alpha_bar(img_size=img_size, shift=True)
    assert torch.all(shifted < base)


def test_the_snr_ratio_is_the_square_of_the_resolution_ratio():
    """Hoogeboom et al.: SNR_shifted = SNR * (ref / res)^2."""
    base = alpha_bar().double()
    shifted = alpha_bar(img_size=1024, shift=True).double()

    snr_base = base / (1 - base)
    snr_shifted = shifted / (1 - shifted)
    expected = (64.0 / 1024.0) ** 2

    assert torch.allclose(snr_shifted / snr_base, torch.full_like(base, expected), rtol=1e-5)


def test_shifting_by_a_smaller_resolution_makes_the_schedule_quieter():
    """The map is not one-directional; below the reference it adds signal back."""
    assert torch.all(alpha_bar(img_size=32, shift=True) > alpha_bar())


# ---------------------------------------------------------------------------
# The resulting schedule is still a valid diffusion process
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_shifted_schedules_stay_valid_for_every_base_schedule(schedule):
    cfg = DiffusionConfig(
        n_steps=1000, schedule_type=schedule, shift_schedule=True, img_size=1024
    )
    assert torch.all(cfg.alpha_cumulative > 0) and torch.all(cfg.alpha_cumulative < 1)
    assert torch.all(cfg.alpha_cumulative[1:] <= cfg.alpha_cumulative[:-1])
    assert torch.all(cfg.beta > 0) and torch.all(cfg.beta < 1)
    assert torch.all(torch.isfinite(cfg.sqrt_one_minus_alpha_cumulative))


def test_beta_and_alpha_describe_the_shifted_process_not_the_base_one():
    """DDPM registers beta and alpha as buffers, so they must be recomputed."""
    cfg = DiffusionConfig(
        n_steps=1000, schedule_type="linear", shift_schedule=True, img_size=1024
    )
    reconstructed = torch.cumprod(1 - cfg.beta, dim=0)
    assert torch.allclose(reconstructed, cfg.alpha_cumulative, rtol=1e-4, atol=1e-6)


def test_the_final_step_actually_destroys_the_signal_at_1024_px():
    """The high-noise end is the end that carries global layout.

    Under the standard schedule a 1024 px cell still retains signal at t = T.
    """
    base = alpha_bar()[-1].item()
    shifted = alpha_bar(img_size=1024, shift=True)[-1].item()
    assert shifted < base / 100


# ---------------------------------------------------------------------------
# Effect on q_sample, which is where it reaches training
# ---------------------------------------------------------------------------


def test_q_sample_keeps_less_of_the_cell_under_the_shifted_schedule():
    torch.manual_seed(0)
    x_0 = torch.randn(4, 6, 16, 16)
    eps = torch.randn_like(x_0)
    t = torch.full((4,), 500, dtype=torch.long)

    def retained(cfg):
        model = DDPM(1000, torch.nn.Identity(), diffusion_config=cfg)
        x_t = model.q_sample(x_0, t, eps)
        # Correlation between the noised cell and the clean one.
        return torch.nn.functional.cosine_similarity(
            x_t.flatten(1), x_0.flatten(1), dim=1
        ).mean().item()

    base = retained(DiffusionConfig(1000, "linear"))
    shifted = retained(
        DiffusionConfig(1000, "linear", shift_schedule=True, img_size=1024)
    )
    assert shifted < base


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------


def test_shift_without_an_img_size_fails_loudly():
    with pytest.raises(ValueError, match="img_size"):
        DiffusionConfig(1000, "linear", shift_schedule=True)


def test_shift_is_off_by_default_so_existing_runs_are_unchanged():
    assert torch.allclose(DiffusionConfig(1000, "linear").alpha_cumulative, alpha_bar())
    assert TrainingConfig().shift_schedule is False
    assert TrainingConfig().ref_img_size == 64


def test_training_config_carries_the_shift_through_yaml(tmp_path):
    import yaml

    path = tmp_path / "cfg.yaml"
    path.write_text(
        yaml.dump({"img_size": 1024, "shift_schedule": True, "ref_img_size": 64,
                   "device": "cpu"}),
        encoding="utf-8",
    )
    cfg = TrainingConfig.from_yaml(str(path))
    assert cfg.shift_schedule and cfg.img_size == 1024

    diffusion = DiffusionConfig(
        n_steps=cfg.timesteps,
        schedule_type=cfg.schedule_type,
        shift_schedule=cfg.shift_schedule,
        img_size=cfg.img_size,
        ref_img_size=cfg.ref_img_size,
    )
    assert diffusion.alpha_cumulative[-1] < alpha_bar()[-1]


def test_to_dict_round_trips_the_shift():
    cfg = DiffusionConfig(1000, "linear", shift_schedule=True, img_size=512)
    rebuilt = DiffusionConfig.from_dict(cfg.to_dict())
    assert torch.allclose(rebuilt.alpha_cumulative, cfg.alpha_cumulative)
