"""
Tests for diffusion schedule types.

Tests all schedule types (linear, fixed, vp, ve) to ensure:
1. Beta sequences are valid (in range [0, 1])
2. Alpha_cumulative is monotonically decreasing
3. DDPM can be initialized with each schedule
4. Schedules are deterministic
"""

import torch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.diffusion_config import DiffusionConfig, ScheduleType
from model.ddpm import DDPM
from model.unet import DDPMUnet


def test_schedule_validity(config: DiffusionConfig, schedule_name: str):
    """Test that beta values are valid."""
    print(f"\n  Testing {schedule_name} schedule validity...")

    # Check beta range
    assert torch.all(config.beta >= 0), f"{schedule_name}: Beta contains negative values"
    assert torch.all(config.beta <= 1), f"{schedule_name}: Beta contains values > 1"

    # Check no NaN or Inf
    assert not torch.any(torch.isnan(config.beta)), f"{schedule_name}: Beta contains NaN"
    assert not torch.any(torch.isinf(config.beta)), f"{schedule_name}: Beta contains Inf"

    # Check alpha_cumulative is monotonically decreasing
    alpha_cumulative = config.alpha_cumulative
    assert torch.all(alpha_cumulative[1:] <= alpha_cumulative[:-1]), \
        f"{schedule_name}: alpha_cumulative is not monotonically decreasing"

    # Check alpha_cumulative range
    assert torch.all(alpha_cumulative >= 0), f"{schedule_name}: alpha_cumulative contains negative"
    assert torch.all(alpha_cumulative <= 1), f"{schedule_name}: alpha_cumulative contains values > 1"

    print(f"    [OK] Beta range: [{config.beta.min():.6f}, {config.beta.max():.6f}]")
    print(f"    [OK] Alpha_bar range: [{alpha_cumulative.min():.6f}, {alpha_cumulative.max():.6f}]")
    return True


def test_ddpm_initialization(schedule_type: str, **kwargs):
    """Test that DDPM can be initialized with the schedule."""
    print(f"\n  Testing DDPM initialization with {schedule_type}...")

    config = DiffusionConfig(
        n_steps=1000,
        schedule_type=schedule_type,
        device="cpu",
        **kwargs
    )

    # Create U-Net and DDPM
    unet = DDPMUnet(1, 32)  # Smaller model for testing
    ddpm = DDPM(1000, unet, diffusion_config=config)

    # Verify buffers are registered
    assert hasattr(ddpm, 'beta'), "DDPM missing beta buffer"
    assert hasattr(ddpm, 'alpha'), "DDPM missing alpha buffer"
    assert hasattr(ddpm, 'alpha_bar'), "DDPM missing alpha_bar buffer"

    # Verify shapes
    assert ddpm.beta.shape == (1000,), f"Beta shape mismatch: {ddpm.beta.shape}"
    assert ddpm.alpha.shape == (1000,), f"Alpha shape mismatch: {ddpm.alpha.shape}"
    assert ddpm.alpha_bar.shape == (1000,), f"Alpha_bar shape mismatch: {ddpm.alpha_bar.shape}"

    print(f"[OK] DDPM initialized successfully")
    return True


def test_schedule_determinism(schedule_type: str, **kwargs):
    """Test that schedules are deterministic."""
    print(f"\n  Testing {schedule_type} schedule determinism...")

    config1 = DiffusionConfig(n_steps=1000, schedule_type=schedule_type, device="cpu", **kwargs)
    config2 = DiffusionConfig(n_steps=1000, schedule_type=schedule_type, device="cpu", **kwargs)

    assert torch.allclose(config1.beta, config2.beta), f"{schedule_type}: Schedule is not deterministic"
    print(f"[OK] Schedule is deterministic")
    return True


def run_all_tests():
    """Run all schedule tests."""
    print("=" * 60)
    print("SCHEDULE TESTS")
    print("=" * 60)

    # Define schedule configurations (using research repo naming)
    schedule_configs = {
        "linear": {"b0": 1e-4, "bT": 2e-2},
        "fixed": {"beta": 6e-3},
        "vp": {"beta_d": 19.9, "beta_min": 0.1},
        "ve": {"sigma_min": 0.02, "sigma_max": 100.0},
    }

    results = {}

    for schedule_name, kwargs in schedule_configs.items():
        print(f"\n{'=' * 60}")
        print(f"Testing: {schedule_name.upper()}")
        print(f"{'=' * 60}")

        try:
            # Create config
            config = DiffusionConfig(n_steps=1000, schedule_type=schedule_name, device="cpu", **kwargs)
            config.print_summary()

            # Run tests
            test_schedule_validity(config, schedule_name)
            test_ddpm_initialization(schedule_name, **kwargs)
            test_schedule_determinism(schedule_name, **kwargs)

            results[schedule_name] = "PASS"

        except Exception as e:
            print(f"\n[FAIL] FAILED: {e}")
            results[schedule_name] = f"FAIL: {e}"

    # Summary
    print(f"\n{'=' * 60}")
    print("TEST SUMMARY")
    print(f"{'=' * 60}")
    for schedule, result in results.items():
        status = "[PASS]" if result == "PASS" else "[FAIL]"
        print(f"  {schedule:10s}: {status}")

    all_passed = all(r == "PASS" for r in results.values())
    print(f"\n{'=' * 60}")
    if all_passed:
        print("ALL TESTS PASSED [OK]")
    else:
        print("SOME TESTS FAILED [FAIL]")
    print(f"{'=' * 60}")

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
