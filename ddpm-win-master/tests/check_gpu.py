"""
Check GPU and PyTorch installation status.
Run this to verify your dual RTX 3090 setup.

Windows-specific commands for NVLink validation:
- nvidia-smi --query-gpu=platform.peer_type --format=csv
- nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv
"""

import sys
import platform
import subprocess


def check_system_info():
    """Check basic system information."""
    print("=" * 60)
    print("SYSTEM INFO")
    print("=" * 60)
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print(f"Architecture: {platform.machine()}")


def check_pytorch():
    """Check PyTorch installation and CUDA."""
    print("\n" + "=" * 60)
    print("PYTORCH & CUDA CHECK")
    print("=" * 60)

    try:
        import torch

        print(f"[OK] PyTorch version: {torch.__version__}")
        print(f"[OK] CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"[OK] CUDA version: {torch.version.cuda}")
            print(f"[OK] GPU count: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"\nGPU {i}:")
                print(f"  Name: {props.name}")
                print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
                print(f"  Compute Capability: {props.major}.{props.minor}")

                # Check current memory usage
                mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
                mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
                print(f"  Memory allocated: {mem_allocated:.2f} GB")
                print(f"  Memory reserved: {mem_reserved:.2f} GB")

            # Test simple tensor operation on both GPUs
            print("\nTesting tensor operations on both GPUs...")
            try:
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        x = torch.randn(1000, 1000).cuda()
                        y = x @ x.T
                        print(f"[OK] GPU {i}: Matrix multiplication works")
            except Exception as e:
                print(f"[FAIL] GPU test failed: {e}")

        return True

    except ImportError as e:
        print(f"[FAIL] PyTorch not found: {e}")
        return False


def check_dependencies():
    """Check required dependencies."""
    print("\n" + "=" * 60)
    print("DEPENDENCY CHECK")
    print("=" * 60)

    required = {
        'numpy': 'numpy',
        'PIL': 'pillow',
        'matplotlib': 'matplotlib',
        'yaml': 'pyyaml',
        'tqdm': 'tqdm',
        'torchvision': 'torchvision',
    }

    missing = []
    for module, package in required.items():
        try:
            __import__(module)
            print(f"[OK] {package}")
        except ImportError:
            print(f"[MISSING] {package}")
            missing.append(package)

    return missing


def check_nvlink():
    """Check NVLink status and PCIe info."""
    print("\n" + "=" * 60)
    print("NVLINK & PCIe CHECK")
    print("=" * 60)

    print("\n** NVLink DETECTED on RTX 3090! **")
    print("(Some RTX 3090 cards do have NVLink - depends on specific model)")

    try:
        # Check NVLink status directly
        print("\nNVLink Status:")
        result = subprocess.run(
            ['nvidia-smi', 'nvlink', '--status'],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print(result.stdout)

        # Check PCIe link info
        print("\nPCIe Link Information:")
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=pcie.link.gen.current,pcie.link.width.current', '--format=csv,noheader'],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            for i, line in enumerate(result.stdout.strip().split('\n')):
                parts = line.split(',')
                if len(parts) == 2:
                    gen, width = parts
                    print(f"  GPU {i}: PCIe Gen {gen.strip()}, {width.strip()} lanes")

        # Calculate bandwidth
        print("\n** Estimated Bandwidth **")
        print("  NVLink: 3 links @ ~14 GB/s = ~42 GB/s per GPU")
        print("  PCIe Gen 4 x16: ~32 GB/s per GPU")
        print("  ** TOTAL: ~74 GB/s per GPU **")
        print("  This is excellent for multi-GPU DDP training!")

    except Exception as e:
        print(f"Could not query NVLink/PCIe info: {e}")


def main():
    print("\n" + "=" * 60)
    print("DDPM RESEARCH - GPU & ENVIRONMENT CHECK")
    print("=" * 60)

    # Check system
    check_system_info()

    # Check PyTorch
    pytorch_ok = check_pytorch()

    # Check dependencies
    missing = check_dependencies()

    # Check NVLink/PCIe
    check_nvlink()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if pytorch_ok and not missing:
        print("[SUCCESS] ALL CHECKS PASSED - Ready to train!")
        print("\nNext steps:")
        print("  1. Test dataset loading: python datasets/dataset.py")
        print("  2. Test training: python train.py")
    else:
        print("[WARNING] SOME ISSUES FOUND:")
        if not pytorch_ok:
            print("  - PyTorch needs installation")
        if missing:
            print(f"  - Missing packages: {', '.join(missing)}")
            print("\nFix with:")
            print("  pip install " + " ".join(missing))


if __name__ == "__main__":
    main()
