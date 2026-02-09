"""
Final test suite - runs all tests before moving to next phase.
Validates the complete foundation before Phase 2.2 (Beta Schedules).
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_test(test_path: str, test_name: str) -> bool:
    """Run a test file and return success status."""
    print("\n" + "=" * 60)
    print(f"Running: {test_name}")
    print("=" * 60)

    import subprocess
    result = subprocess.run(
        [sys.executable, test_path],
        capture_output=False,
        text=True
    )

    if result.returncode == 0:
        print(f"\n[OK] {test_name} PASSED")
        return True
    else:
        print(f"\n[FAIL] {test_name} FAILED")
        return False


def main():
    print("=" * 60)
    print("FINAL TEST SUITE - PRE-PHASE 2.2")
    print("=" * 60)
    print("\nRunning all tests to validate foundation...")
    print("Tests located in: tests/\n")

    tests = [
        ("tests/test_config_yaml.py", "Config YAML Loading"),
        ("tests/test_train_config.py", "Train with Config (Single & Dual GPU)"),
    ]

    results = []
    for test_path, test_name in tests:
        success = run_test(test_path, test_name)
        results.append((test_name, success))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUITE SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, success in results:
        status = "[OK]" if success else "[FAIL]"
        print(f"{status} {test_name}")
        if not success:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n[SUCCESS] ALL TESTS PASSED!")
        print("\nFoundation validated. Ready for Phase 2.2 (Beta Schedules).")
        print("\nNext steps:")
        print("  - Phase 2.2: Beta Schedules (VP/VE/EDM variants)")
        print("  - Phase 2.3: Checkpointing System")
        print("  - Phase 2.4: Training Loop Integration")
        return 0
    else:
        print("\n[FAIL] Some tests failed. Fix issues before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
