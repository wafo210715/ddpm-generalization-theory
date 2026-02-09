"""
Test configuration loading from YAML files.
Validates TrainingConfig and DiffusionConfig with YAML.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, '.')

from config_loader import TrainingConfig
from model.diffusion_config import DiffusionConfig


def test_training_config_yaml():
    """Test loading TrainingConfig from YAML."""
    print("\n" + "=" * 60)
    print("TEST: TrainingConfig YAML Loading")
    print("=" * 60)

    # Test loading default config
    print("\n1. Loading configs/default.yaml...")
    config = TrainingConfig.from_yaml("configs/default.yaml")
    config.print_summary()

    # Verify values
    assert config.dataset_name == "mnist", "Dataset name mismatch"
    assert config.img_size == 32, "Image size mismatch"
    assert config.batch_size == 128, "Batch size mismatch"
    assert config.n_steps == 50000, "Training steps mismatch"

    # Test checkpoint schedule
    checkpoints = config.get_all_checkpoints()
    print(f"\n2. Checkpoint schedule (first 10): {checkpoints[:10]}")
    assert len(checkpoints) > 0, "Checkpoint schedule empty"

    # Test geospatial config
    print("\n3. Loading configs/geospatial.yaml...")
    geo_config = TrainingConfig.from_yaml("configs/geospatial.yaml")
    geo_config.print_summary()

    assert geo_config.dataset_name == "geospatial", "Geo dataset name mismatch"
    assert geo_config.img_size == 512, "Geo image size mismatch"
    assert geo_config.batch_size == 16, "Geo batch size mismatch"

    print("\n[OK] TrainingConfig YAML loading PASSED!")
    return config


def test_diffusion_config():
    """Test DiffusionConfig with different schedules."""
    print("\n" + "=" * 60)
    print("TEST: DiffusionConfig Schedule Types")
    print("=" * 60)

    # Test linear schedule
    print("\n1. Linear schedule (default)...")
    linear_config = DiffusionConfig(n_steps=1000, schedule_type="linear", device="cpu")
    linear_config.print_summary()

    # Test fixed schedule
    print("\n2. Fixed schedule...")
    fixed_config = DiffusionConfig(n_steps=1000, schedule_type="fixed", device="cpu", beta=6e-3)
    fixed_config.print_summary()

    # Test VP schedule
    print("\n3. VP schedule...")
    vp_config = DiffusionConfig(n_steps=1000, schedule_type="vp", device="cpu")
    vp_config.print_summary()

    # Test VE schedule
    print("\n4. VE schedule...")
    ve_config = DiffusionConfig(n_steps=1000, schedule_type="ve", device="cpu")
    ve_config.print_summary()

    # Verify derived quantities
    print("\n5. Verifying derived quantities...")
    assert linear_config.alpha.shape == (1000,), "Alpha shape mismatch"
    assert linear_config.alpha_cumulative.shape == (1000,), "Alpha_cumulative shape mismatch"
    assert linear_config.sqrt_alpha_cumulative.shape == (1000,), "Sqrt_alpha_cumulative shape mismatch"

    print(f"   Beta shape: {linear_config.beta.shape}")
    print(f"   Alpha shape: {linear_config.alpha.shape}")
    print(f"   Alpha_cumulative range: [{linear_config.alpha_cumulative.min():.6f}, {linear_config.alpha_cumulative.max():.6f}]")

    print("\n[OK] DiffusionConfig PASSED!")
    return linear_config


def test_config_integration():
    """Test TrainingConfig and DiffusionConfig integration."""
    print("\n" + "=" * 60)
    print("TEST: Config Integration")
    print("=" * 60)

    # Load training config
    print("\n1. Loading training config...")
    train_config = TrainingConfig.from_yaml("configs/default.yaml")

    # Create diffusion config
    print("\n2. Creating diffusion config...")
    diff_config = DiffusionConfig(
        n_steps=train_config.timesteps,
        schedule_type="linear",
        device=train_config.device
    )

    # Verify timesteps match
    assert train_config.timesteps == diff_config.n_steps, "Timesteps mismatch"
    print(f"\n3. Timesteps match: {train_config.timesteps} == {diff_config.n_steps}")

    print("\n[OK] Config integration PASSED!")
    return train_config, diff_config


def test_save_config():
    """Test saving config to YAML."""
    print("\n" + "=" * 60)
    print("TEST: Save Config to YAML")
    print("=" * 60)

    # Create config
    print("\n1. Creating test config...")
    config = TrainingConfig(
        dataset_name="test",
        img_size=64,
        batch_size=32,
        n_steps=1000
    )

    # Save to YAML
    print("\n2. Saving to configs/test_output.yaml...")
    config.to_yaml("configs/test_output.yaml")

    # Load back
    print("\n3. Loading back from YAML...")
    loaded_config = TrainingConfig.from_yaml("configs/test_output.yaml")

    # Verify
    assert loaded_config.dataset_name == "test", "Dataset name mismatch after save/load"
    assert loaded_config.img_size == 64, "Image size mismatch after save/load"
    assert loaded_config.batch_size == 32, "Batch size mismatch after save/load"

    print("\n4. Cleaning up test file...")
    os.remove("configs/test_output.yaml")

    print("\n[OK] Save config to YAML PASSED!")


def main():
    print("=" * 60)
    print("CONFIGURATION TEST SUITE")
    print("=" * 60)

    try:
        # Run all tests
        test_training_config_yaml()
        test_diffusion_config()
        test_config_integration()
        test_save_config()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print("\nConfiguration system is ready for use.")
        print("Next: Update train.py to use config loading.")

    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
