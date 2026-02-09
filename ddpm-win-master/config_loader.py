"""
Configuration loader for DDPM training.
Load configs from YAML files with sensible defaults.
"""

import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import os
import yaml


@dataclass
class TrainingConfig:
    """Training configuration with step-based training."""

    # Dataset
    dataset_name: str = "mnist"
    img_channels: int = 1
    img_size: int = 32
    batch_size: int = 128

    # Training (step-based, not epoch-based)
    lr: float = 1e-4
    n_steps: int = 50000
    timesteps: int = 1000
    log_every: int = 100

    # Device
    device: str = "cuda"  # Will auto-detect
    use_dual_gpu: bool = False  # Enable dual GPU training with DataParallel

    # Paths
    checkpoint_dir: str = "model_ckpts"
    log_dir: str = "logs"
    run_name: Optional[str] = None  # Auto-generated if None

    # Data normalization
    center: bool = True
    normalize: bool = False

    # Diffusion schedule configuration
    schedule_type: str = "linear"  # Options: linear, fixed, vp, ve
    # Linear schedule parameters (DDPM standard naming)
    b0: float = 1e-4  # Starting beta
    bT: float = 2e-2  # Ending beta
    # Fixed schedule parameters
    beta: float = 6e-3  # Constant beta value
    # VP schedule parameters (Song et al. SDE naming)
    beta_d: float = 19.9  # Extent of noise level schedule
    beta_min: float = 0.1  # Initial slope
    # VE schedule parameters (Song et al. SDE naming)
    sigma_min: float = 0.02  # Minimum noise level
    sigma_max: float = 100.0  # Maximum noise level

    # Checkpointing (for generalization window)
    save_initial: List[int] = field(default_factory=lambda: [100, 250, 500, 1000, 2500, 5000])
    save_log: List[int] = field(default_factory=list)  # Auto-generated if empty
    save_interval: int = 5000

    def __post_init__(self):
        """Post-processing after initialization."""
        # Auto-detect device
        if self.device == "cuda":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Generate run name if not provided
        if self.run_name is None:
            self.run_name = f"{self.dataset_name}_{self.img_size}x{self.img_size}_steps{self.n_steps}"

        # Set default log schedule if not provided
        if not self.save_log:
            self.save_log = self._generate_log_schedule(10000, self.n_steps)

    def _generate_log_schedule(self, start: int, end: int) -> List[int]:
        """Generate logarithmic checkpoint schedule."""
        schedule = []
        current = start
        while current <= end:
            schedule.append(current)
            if current < 100000:
                current *= 2
            else:
                current = int(current * 1.5)
        return schedule

    def get_all_checkpoints(self) -> List[int]:
        """Get all checkpoint steps combined."""
        return sorted(set(self.save_initial + self.save_log))

    @classmethod
    def from_yaml(cls, path: str) -> 'TrainingConfig':
        """Load config from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary."""
        return cls(**config_dict)

    def to_yaml(self, path: str):
        """Save config to YAML file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    def print_summary(self):
        """Print configuration summary for validation."""
        print("=" * 50)
        print("TRAINING CONFIGURATION")
        print("=" * 50)
        print(f"Dataset: {self.dataset_name}")
        print(f"Image: {self.img_channels} channels, {self.img_size}x{self.img_size}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.lr}")
        print(f"Training steps: {self.n_steps}")
        print(f"Diffusion timesteps: {self.timesteps}")
        print(f"Device: {self.device}")
        print(f"Dual GPU: {self.use_dual_gpu}")
        print(f"Run name: {self.run_name}")
        print(f"\nCheckpoint schedule: {self.get_all_checkpoints()[:10]}...")
        print("=" * 50)
