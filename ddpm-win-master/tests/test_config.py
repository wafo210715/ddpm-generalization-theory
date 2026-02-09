"""
Test script for config_loader.py
Run this to validate the config system works.
"""

import sys
sys.path.insert(0, '.')

from config_loader import TrainingConfig


def test_default_config():
    """Test 1: Default configuration"""
    print("\n" + "="*60)
    print("TEST 1: Default Configuration")
    print("="*60)

    config = TrainingConfig()
    config.print_summary()

    # Validate key fields
    assert config.dataset_name == "mnist"
    assert config.n_steps == 50000
    assert config.device in ["cuda", "cpu"]
    assert config.run_name is not None

    print("\n✓ Test 1 PASSED: Default config works\n")


def test_custom_config():
    """Test 2: Custom configuration"""
    print("="*60)
    print("TEST 2: Custom Configuration")
    print("="*60)

    config = TrainingConfig(
        dataset_name="geospatial",
        img_channels=3,
        img_size=512,
        batch_size=16,
        n_steps=100000
    )
    config.print_summary()

    # Validate
    assert config.dataset_name == "geospatial"
    assert config.img_size == 512
    assert config.batch_size == 16
    assert config.run_name == "geospatial_512x512_steps100000"

    print("\n✓ Test 2 PASSED: Custom config works\n")


def test_checkpoint_schedule():
    """Test 3: Logarithmic checkpoint schedule"""
    print("="*60)
    print("TEST 3: Checkpoint Schedule Generation")
    print("="*60)

    config = TrainingConfig(n_steps=50000)
    checkpoints = config.get_all_checkpoints()

    print(f"\nTotal checkpoints: {len(checkpoints)}")
    print(f"First 10 checkpoints: {checkpoints[:10]}")
    print(f"Last 5 checkpoints: {checkpoints[-5:]}")

    # Validate schedule
    assert 100 in checkpoints
    assert 5000 in checkpoints
    assert checkpoints == sorted(checkpoints)  # Should be sorted

    print("\n✓ Test 3 PASSED: Checkpoint schedule generated correctly\n")


if __name__ == "__main__":
    try:
        test_default_config()
        test_custom_config()
        test_checkpoint_schedule()

        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
