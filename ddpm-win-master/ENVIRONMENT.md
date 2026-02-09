# Environment Setup

## Virtual Environment (uv)

### Initial Setup
```bash
# Install uv (if not already installed)
pip install uv

# Create virtual environment
uv venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies from lock file (reproducible)
uv pip sync uv.lock
```

### Virtual Environment Activation (Windows)

The activation command above works in Command Prompt (cmd.exe). If you're using PowerShell and get an "UnauthorizedAccess" error, use one of these solutions:

#### Option A: Use Command Prompt (Recommended)
```cmd
cd C:\sd\ddpm-generalization-theory\ddpm-win-master
.venv\Scripts\activate.bat
```

#### Option B: Change PowerShell Execution Policy (Run as Administrator once)
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Then activate normally:
```powershell
cd C:\sd\ddpm-generalization-theory\ddpm-win-master
.venv\Scripts\Activate.ps1
```

#### Option C: Bypass Without Changing Policy
```powershell
cd C:\sd\ddpm-generalization-theory\ddpm-win-master
powershell -ExecutionPolicy Bypass -File .venv\Scripts\Activate.ps1
```

#### Option D: Use Python Directly (No Activation)
```powershell
.venv\Scripts\python.exe train.py
```

### Adding New Dependencies
```bash
# Add a new package
uv pip add <package-name>

# Update lock file after manual changes to requirements.txt
uv pip compile requirements.txt -o uv.lock
```

### Lock File
- **uv.lock**: Pinned versions for reproducible builds
- Run `uv pip sync uv.lock` to install exact versions

## GPU Configuration

**Platform**: Windows 10
**GPUs**: 2x RTX 3090 with NVLink
- GPU 0: 3x NVLink @ 14.062 GB/s + PCIe Gen 4 x16 @ ~32 GB/s
- GPU 1: 3x NVLink @ 14.062 GB/s + PCIe Gen 4 x16 @ ~32 GB/s
- **Total**: ~74 GB/s interconnect per GPU

**Multi-GPU Training** (proven to work on this machine):
```python
from torch.distributed import DistributedDataParallel as DDP

strategy = DDPStrategy(process_group_backend='gloo', find_unused_parameters=True)
```

## Validation Commands

```bash
# Check GPU and NVLink status
nvidia-smi nvlink --status

# Run environment check
python check_gpu.py

# Test dataset loading
python datasets/dataset.py
```

## Current Dependencies

See `uv.lock` for exact pinned versions.
