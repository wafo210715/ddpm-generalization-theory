"""
Test updated train.py with config support and dual GPU.
Quick test with 20 steps to verify everything works.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from config_loader import TrainingConfig
from train import train_with_config, save_model


def test_train_single_gpu():
    """Test training with config on single GPU."""
    print("\n" + "=" * 60)
    print("TEST: Train with Config (Single GPU)")
    print("=" * 60)

    # Create test config
    config = TrainingConfig(
        dataset_name="mnist",
        img_channels=1,
        img_size=32,
        batch_size=128,
        lr=1e-4,
        n_steps=20,  # Quick test
        timesteps=1000,
        log_every=5,
        use_dual_gpu=False,
        checkpoint_dir="model_ckpts/test"
    )

    print("\nRunning 20-step training test...")
    train_with_config(config)

    print("\n[OK] Single GPU training test PASSED!")
    return config


def test_train_dual_gpu():
    """Test training with config on dual GPU."""
    print("\n" + "=" * 60)
    print("TEST: Train with Config (Dual GPU)")
    print("=" * 60)

    # Check dual GPU availability
    if torch.cuda.device_count() < 2:
        print("\n[SKIP] Dual GPU test skipped (only 1 GPU available)")
        return None

    # Create test config with dual GPU enabled
    config = TrainingConfig(
        dataset_name="mnist",
        img_channels=1,
        img_size=32,
        batch_size=256,  # Larger batch for dual GPU
        lr=1e-4,
        n_steps=20,  # Quick test
        timesteps=1000,
        log_every=5,
        use_dual_gpu=True,
        checkpoint_dir="model_ckpts/test"
    )

    print("\nRunning 20-step dual GPU training test...")
    train_with_config(config)

    print("\n[OK] Dual GPU training test PASSED!")
    return config


def test_config_loading():
    """Test loading config from YAML."""
    print("\n" + "=" * 60)
    print("TEST: Config Loading from YAML")
    print("=" * 60)

    # Load default config
    print("\n1. Loading configs/default.yaml...")
    config = TrainingConfig.from_yaml("configs/default.yaml")
    config.print_summary()
    print(f"  use_dual_gpu: {config.use_dual_gpu}")

    # Load geospatial config
    print("\n2. Loading configs/geospatial.yaml...")
    geo_config = TrainingConfig.from_yaml("configs/geospatial.yaml")
    geo_config.print_summary()
    print(f"  use_dual_gpu: {geo_config.use_dual_gpu}")
    print(f"  img_size: {geo_config.img_size}")

    print("\n[OK] Config loading test PASSED!")
    return config


def main():
    print("=" * 60)
    print("TRAIN CONFIG TEST SUITE")
    print("=" * 60)

    try:
        # Test 1: Config loading
        test_config_loading()

        # Test 2: Single GPU training
        test_train_single_gpu()

        # Test 3: Dual GPU training (if available)
        test_train_dual_gpu()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print("\nUpdated train.py with config is ready for use.")
        print("Usage:")
        print("  python train.py                           # Default config")
        print("  python train.py --config configs/default.yaml")
        print("  python train.py --config configs/geospatial.yaml")

    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
