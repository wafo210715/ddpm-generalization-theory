"""
Diffusion configuration with multiple beta schedule types.
Supports linear, fixed, VP, and VE schedules for DDPM training.
Note: Cosine schedule is implemented but has numerical instability issues.
"""

import torch
from enum import Enum
from typing import Optional, Dict, Any


class ScheduleType(Enum):
    """Supported beta schedule types."""
    LINEAR = "linear"
    FIXED = "fixed"
    VP = "vp"      # Variance Preserving (from Song et al. SDE paper)
    VE = "ve"      # Variance Exploding (from Song et al. SDE paper)


class DiffusionConfig:
    """
    Diffusion process configuration with multiple schedule types.

    Supports:
    - Linear schedule (original DDPM) - RECOMMENDED
    - VP schedule (Variance Preserving, from Song et al. SDE paper, discretized) - RECOMMENDED
    - VE schedule (Variance Exploding, from Song et al. SDE paper, discretized) - RECOMMENDED
    - Fixed schedule (constant beta)
    """

    def __init__(
        self,
        n_steps: int = 1000,
        schedule_type: str = "linear",
        device: str = "cpu",
        **schedule_kwargs
    ):
        self.n_steps = n_steps
        self.schedule_type = schedule_type
        self.device = device
        self.schedule_kwargs = schedule_kwargs
        self.initialize()

    def initialize(self):
        """Initialize diffusion parameters based on schedule type."""
        if self.schedule_type == ScheduleType.LINEAR.value:
            self.beta = self.linear_schedule(**self.schedule_kwargs)
        elif self.schedule_type == ScheduleType.FIXED.value:
            self.beta = self.fixed_schedule(**self.schedule_kwargs)
        elif self.schedule_type == ScheduleType.VP.value:
            self.beta = self.vp_schedule(**self.schedule_kwargs)
        elif self.schedule_type == ScheduleType.VE.value:
            self.beta = self.ve_schedule(**self.schedule_kwargs)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule_type}")

        # Move to device
        self.beta = self.beta.to(self.device)

        # Derived quantities
        self.alpha = 1 - self.beta
        self.alpha_cumulative = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_cumulative = torch.sqrt(self.alpha_cumulative)
        self.one_by_sqrt_alpha = 1. / torch.sqrt(self.alpha)
        self.sqrt_one_minus_alpha_cumulative = torch.sqrt(1 - self.alpha_cumulative)

    def linear_schedule(self, b0: float = 1e-4, bT: float = 2e-2) -> torch.Tensor:
        """
        Linear DDPM schedule (original).

        Args:
            b0: Starting beta value
            bT: Ending beta value
        """
        scale = 1000 / self.n_steps
        beta_start = scale * b0
        beta_end = scale * bT
        return torch.linspace(beta_start, beta_end, self.n_steps, dtype=torch.float32)

    def fixed_schedule(self, beta: float = 6e-3) -> torch.Tensor:
        """
        Fixed beta schedule (constant beta).

        Args:
            beta: Constant beta value
        """
        return torch.full((self.n_steps,), beta, dtype=torch.float32)

    def vp_schedule(self, beta_d: float = 19.9, beta_min: float = 0.1) -> torch.Tensor:
        """
        Variance Preserving (VP) schedule from Song et al. SDE paper.

        Discretized for DDPM: converts continuous sigma(t) to discrete beta.

        Original formulation:
            sigma(t) = sqrt(exp(0.5 * beta_d * t^2 + beta_min * t) - 1)

        Args:
            beta_d: Extent of the noise level schedule (default 19.9)
            beta_min: Initial slope of the noise level schedule (default 0.1)
        """
        # Normalized time t ∈ [0, 1] for discrete timesteps
        t = torch.linspace(0, 1, self.n_steps, dtype=torch.float32)

        # Compute sigma at each timestep (continuous VP formula)
        sigma = torch.sqrt(torch.exp(0.5 * beta_d * (t ** 2) + beta_min * t) - 1)

        # Convert sigma to alpha_bar: alpha_bar = 1 / (1 + sigma^2)
        alpha_bar = 1.0 / (1.0 + sigma ** 2)

        # Convert alpha_bar to beta: beta_t = 1 - (alpha_bar_t / alpha_bar_{t-1})
        # Handle first timestep separately
        alphas = torch.zeros_like(alpha_bar)
        alphas[0] = alpha_bar[0]
        alphas[1:] = alpha_bar[1:] / alpha_bar[:-1]
        beta = 1.0 - alphas

        # Clip to valid range
        beta = torch.clip(beta, 0.0001, 0.9999)
        return beta

    def ve_schedule(self, sigma_min: float = 0.02, sigma_max: float = 100.0) -> torch.Tensor:
        """
        Variance Exploding (VE) schedule from Song et al. SDE paper.

        Discretized for DDPM: converts continuous sigma(t) to discrete beta.

        Original formulation:
            sigma(t) = sigma_min * (sigma_max / sigma_min)^t

        Args:
            sigma_min: Minimum noise level (default 0.02)
            sigma_max: Maximum noise level (default 100.0)
        """
        # Normalized time t ∈ [0, 1] for discrete timesteps
        t = torch.linspace(0, 1, self.n_steps, dtype=torch.float32)

        # Compute sigma at each timestep (continuous VE formula)
        sigma = sigma_min * ((sigma_max / sigma_min) ** t)

        # Convert sigma to alpha_bar: alpha_bar = 1 / (1 + sigma^2)
        alpha_bar = 1.0 / (1.0 + sigma ** 2)

        # Convert alpha_bar to beta: beta_t = 1 - (alpha_bar_t / alpha_bar_{t-1})
        alphas = torch.zeros_like(alpha_bar)
        alphas[0] = alpha_bar[0]
        alphas[1:] = alpha_bar[1:] / alpha_bar[:-1]
        beta = 1.0 - alphas

        # Clip to valid range
        beta = torch.clip(beta, 0.0001, 0.9999)
        return beta

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DiffusionConfig':
        """Create from dictionary (for YAML loading)."""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for YAML saving)."""
        return {
            "n_steps": self.n_steps,
            "schedule_type": self.schedule_type,
            "device": self.device,
            **self.schedule_kwargs
        }

    def print_summary(self):
        """Print diffusion configuration summary."""
        print("=" * 50)
        print("DIFFUSION CONFIGURATION")
        print("=" * 50)
        print(f"Schedule type: {self.schedule_type}")
        print(f"Timesteps: {self.n_steps}")
        print(f"Beta range: [{self.beta.min():.6f}, {self.beta.max():.6f}]")
        print(f"Device: {self.device}")
        if self.schedule_kwargs:
            print(f"Schedule params: {self.schedule_kwargs}")
        print("=" * 50)
