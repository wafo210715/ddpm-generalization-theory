"""
Checkpoint Conversion Utility

Convert between PyTorch (.pth) and SafeTensors (.safetensors) formats.

Usage:
    # Convert .pth to .safetensors (for sharing on HuggingFace)
    python scripts/convert_checkpoint.py --input model_ckpts/model.pth

    # Convert .safetensors to .pth
    python scripts/convert_checkpoint.py --input model_ckpts/model.safetensors

    # Specify output path
    python scripts/convert_checkpoint.py --input model.pth --output model.safetensors

    # Convert entire directory
    python scripts/convert_checkpoint.py --input model_ckpts/ --convert-dir
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import torch
except ImportError:
    print("Error: PyTorch not installed. Install with: pip install torch")
    sys.exit(1)

try:
    from safetensors.torch import save_file, load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("Warning: safetensors not installed. Install with: pip install safetensors")


def convert_pth_to_safetensors(input_path, output_path=None, verbose=True):
    """
    Convert a PyTorch .pth checkpoint to SafeTensors format.

    Args:
        input_path: Path to .pth file
        output_path: Path for output .safetensors file (optional, auto-generates if None)
        verbose: Print progress messages

    Returns:
        Path to the converted file
    """
    if not SAFETENSORS_AVAILABLE:
        raise RuntimeError("safetensors library not available. Install with: pip install safetensors")

    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix('.safetensors')
    else:
        output_path = Path(output_path)

    if input_path.suffix != '.pth':
        raise ValueError(f"Input file must be .pth, got: {input_path.suffix}")

    if verbose:
        print(f"Converting: {input_path} -> {output_path}")

    # Load PyTorch checkpoint
    if verbose:
        print("  Loading .pth file...")
    checkpoint = torch.load(input_path, map_location='cpu')

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        # Case 1: state_dict wrapped in 'state_dict' key (common in training scripts)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Extract metadata (non-tensor items)
            metadata = {}
            for k, v in checkpoint.items():
                if k != 'state_dict':
                    # Convert simple types to strings for safetensors metadata
                    if isinstance(v, (str, int, float, bool)):
                        metadata[k] = str(v)
                    elif isinstance(v, (list, tuple)):
                        metadata[k] = str(v)

        # Case 2: raw state_dict (all values are tensors)
        elif all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            state_dict = checkpoint
            metadata = {}

        # Case 3: model wrapped in 'model' key
        elif 'model' in checkpoint and isinstance(checkpoint['model'], dict):
            state_dict = checkpoint['model']
            metadata = {}
            for k, v in checkpoint.items():
                if k != 'model' and isinstance(v, (str, int, float, bool)):
                    metadata[k] = str(v)

        else:
            # Mixed format - extract tensors and separate metadata
            state_dict = {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}
            metadata = {}
            for k, v in checkpoint.items():
                if not isinstance(v, torch.Tensor) and isinstance(v, (str, int, float, bool, list, tuple)):
                    metadata[k] = str(v)

            if verbose and len(metadata) > 0:
                print(f"  Preserved metadata keys: {list(metadata.keys())}")
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)}")

    if verbose:
        print(f"  Found {len(state_dict)} tensor(s)")

    # Save as safetensors
    if verbose:
        print("  Saving as .safetensors...")
    save_file(state_dict, str(output_path), metadata=metadata if metadata else None)

    # Report file sizes
    if verbose:
        input_size = input_path.stat().st_size / (1024 ** 2)  # MB
        output_size = output_path.stat().st_size / (1024 ** 2)  # MB
        size_diff = output_size - input_size
        print(f"  Input size:  {input_size:.2f} MB")
        print(f"  Output size: {output_size:.2f} MB")
        print(f"  Size change: {size_diff:+.2f} MB ({size_diff/input_size*100:+.1f}%)")
        print(f"  Done!")

    return output_path


def convert_safetensors_to_pth(input_path, output_path=None, verbose=True):
    """
    Convert a SafeTensors checkpoint to PyTorch .pth format.

    Args:
        input_path: Path to .safetensors file
        output_path: Path for output .pth file (optional, auto-generates if None)
        verbose: Print progress messages

    Returns:
        Path to the converted file
    """
    if not SAFETENSORS_AVAILABLE:
        raise RuntimeError("safetensors library not available. Install with: pip install safetensors")

    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix('.pth')
    else:
        output_path = Path(output_path)

    if input_path.suffix != '.safetensors':
        raise ValueError(f"Input file must be .safetensors, got: {input_path.suffix}")

    if verbose:
        print(f"Converting: {input_path} -> {output_path}")

    # Load safetensors
    if verbose:
        print("  Loading .safetensors file...")
    tensors = load_file(str(input_path))

    if verbose:
        print(f"  Found {len(tensors)} tensor(s)")

    # Save as PyTorch
    if verbose:
        print("  Saving as .pth...")
    torch.save(tensors, output_path)

    # Report file sizes
    if verbose:
        input_size = input_path.stat().st_size / (1024 ** 2)  # MB
        output_size = output_path.stat().st_size / (1024 ** 2)  # MB
        size_diff = output_size - input_size
        print(f"  Input size:  {input_size:.2f} MB")
        print(f"  Output size: {output_size:.2f} MB")
        print(f"  Size change: {size_diff:+.2f} MB ({size_diff/input_size*100:+.1f}%)")
        print(f"  Done!")

    return output_path


def convert_directory(input_dir, output_dir=None, direction='pth_to_safetensors', verbose=True):
    """
    Convert all checkpoints in a directory.

    Args:
        input_dir: Input directory path
        output_dir: Output directory (optional, creates subdirectory if None)
        direction: 'pth_to_safetensors' or 'safetensors_to_pth'
        verbose: Print progress messages
    """
    input_dir = Path(input_dir)

    if not input_dir.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")

    # Determine output directory
    if output_dir is None:
        if direction == 'pth_to_safetensors':
            output_dir = input_dir / 'safetensors_export'
        else:
            output_dir = input_dir / 'pth_export'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find files to convert
    if direction == 'pth_to_safetensors':
        files = list(input_dir.glob('*.pth'))
        convert_func = convert_pth_to_safetensors
        target_ext = '.safetensors'
    else:
        files = list(input_dir.glob('*.safetensors'))
        convert_func = convert_safetensors_to_pth
        target_ext = '.pth'

    if not files:
        print(f"No {target_ext} files found in {input_dir}")
        return

    if verbose:
        print(f"\nConverting directory: {input_dir}")
        print(f"Found {len(files)} file(s) to convert")
        print(f"Output directory: {output_dir}\n")

    # Convert each file
    success_count = 0
    failed = []

    for i, input_file in enumerate(files, 1):
        try:
            print(f"[{i}/{len(files)}] {input_file.name}")
            output_file = output_dir / (input_file.stem + target_ext)
            convert_func(input_file, output_file, verbose=False)
            success_count += 1
            print(f"  -> OK\n")
        except Exception as e:
            print(f"  -> FAILED: {e}\n")
            failed.append((input_file, str(e)))

    # Summary
    if verbose:
        print(f"Conversion complete: {success_count}/{len(files)} successful")
        if failed:
            print(f"\nFailed conversions:")
            for f, err in failed:
                print(f"  - {f}: {err}")


def auto_convert(input_path, output_path=None, verbose=True):
    """
    Auto-detect input format and convert to opposite format.

    Args:
        input_path: Path to input file
        output_path: Path for output file (optional, auto-generates if None)
        verbose: Print progress messages
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix == '.pth':
        return convert_pth_to_safetensors(input_path, output_path, verbose)
    elif input_path.suffix == '.safetensors':
        return convert_safetensors_to_pth(input_path, output_path, verbose)
    else:
        raise ValueError(f"Unknown file extension: {input_path.suffix}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert between PyTorch (.pth) and SafeTensors (.safetensors) formats',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input file or directory path'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output file or directory path (default: auto-generate)'
    )

    parser.add_argument(
        '--convert-dir',
        action='store_true',
        help='Convert all files in input directory'
    )

    parser.add_argument(
        '--to-safetensors',
        action='store_true',
        help='Force conversion to safetensors format'
    )

    parser.add_argument(
        '--to-pth',
        action='store_true',
        help='Force conversion to .pth format'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress messages'
    )

    args = parser.parse_args()
    verbose = not args.quiet

    try:
        if args.convert_dir:
            # Directory conversion
            direction = 'pth_to_safetensors'
            if args.to_pth:
                direction = 'safetensors_to_pth'
            convert_directory(args.input, args.output, direction, verbose)
        else:
            # Single file conversion
            if args.to_safetensors:
                convert_pth_to_safetensors(args.input, args.output, verbose)
            elif args.to_pth:
                convert_safetensors_to_pth(args.input, args.output, verbose)
            else:
                auto_convert(args.input, args.output, verbose)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
