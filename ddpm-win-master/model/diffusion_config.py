"""
Diffusion configuration with multiple beta schedule types.
Supports linear, fixed, VP, and VE schedules for DDPM training.
Note: Cosine schedule is implemented but has numerical instability issues.

Any of these schedules can additionally be *shifted* for resolution, which the
v4 pipeline map (B2, D1) requires at 1024 px: "the standard DDPM noise schedule
under-noises large images, so it must be shifted".  See
:meth:`DiffusionConfig.shift_alpha_cumulative`.
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
        shift_schedule: bool = False,
        img_size: Optional[int] = None,
        ref_img_size: int = 64,
        **schedule_kwargs
    ):
        """
        Args:
            n_steps: Number of diffusion timesteps.
            schedule_type: One of linear, fixed, vp, ve.
            device: Device the schedule tensors live on.
            shift_schedule: Shift the schedule for resolution (Hoogeboom et al.
                2023, Simple Diffusion; Chen 2023).  Requires img_size.
            img_size: Resolution the schedule is being used at.
            ref_img_size: Resolution the base schedule was tuned for.  Simple
                Diffusion shifts relative to 64x64, which is this default.
            **schedule_kwargs: Parameters of the chosen schedule.
        """
        self.n_steps = n_steps
        self.schedule_type = schedule_type
        self.device = device
        self.shift_schedule = shift_schedule
        self.img_size = img_size
        self.ref_img_size = ref_img_size
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

        # Resolution shift, if asked for.  Applied to alpha_bar, then beta and
        # alpha are recomputed from it, so every downstream quantity — and the
        # buffers DDPM registers — describe the shifted process.
        if self.shift_schedule:
            if self.img_size is None:
                raise ValueError(
                    "shift_schedule=True requires img_size (the resolution the "
                    "schedule is used at)"
                )
            self.alpha_cumulative = self.shift_alpha_cumulative(
                self.alpha_cumulative, self.img_size, self.ref_img_size
            )
            self.alpha = self._alpha_from_alpha_cumulative(self.alpha_cumulative)
            self.beta = 1.0 - self.alpha

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

    @staticmethod
    def shift_alpha_cumulative(
        alpha_cumulative: torch.Tensor,
        img_size: int,
        ref_img_size: int = 64,
    ) -> torch.Tensor:
        """Shift a schedule for resolution (Hoogeboom et al. 2023; Chen 2023).

        A fixed noise level destroys less information in a large image than in
        a small one, because neighbouring pixels are redundant and averaging
        them recovers the signal.  So the standard schedule under-noises at
        1024 px: the model never sees a genuinely destroyed cell, and the
        high-noise end of the trajectory — the end that carries the global
        layout — is never trained.  Simple Diffusion fixes this by dividing the
        signal-to-noise ratio by the square of the resolution ratio:

            SNR(t)         = alpha_bar(t) / (1 - alpha_bar(t))
            SNR_shifted(t) = SNR(t) * (ref_img_size / img_size) ** 2
            alpha_bar_shifted(t) = SNR_shifted(t) / (1 + SNR_shifted(t))

        Args:
            alpha_cumulative: alpha_bar of the base schedule, shape (n_steps,).
            img_size: Resolution the schedule is being used at.
            ref_img_size: Resolution the base schedule was tuned for.

        Returns:
            The shifted alpha_bar.  Monotonically decreasing, in (0, 1), and
            the identity when img_size == ref_img_size.
        """
        if img_size <= 0 or ref_img_size <= 0:
            raise ValueError("img_size and ref_img_size must be positive")

        alpha_bar = alpha_cumulative.to(torch.float64)
        # Guard the endpoints: alpha_bar == 1 makes SNR infinite and
        # alpha_bar == 0 makes it zero, both of which survive the algebra but
        # produce nan through the division.
        alpha_bar = alpha_bar.clamp(1e-12, 1.0 - 1e-12)

        shift = (float(ref_img_size) / float(img_size)) ** 2
        snr = alpha_bar / (1.0 - alpha_bar)
        snr_shifted = snr * shift
        shifted = snr_shifted / (1.0 + snr_shifted)
        return shifted.to(alpha_cumulative.dtype)

    @staticmethod
    def _alpha_from_alpha_cumulative(alpha_cumulative: torch.Tensor) -> torch.Tensor:
        """Per-step alpha implied by a cumulative product, first step included."""
        alpha = torch.empty_like(alpha_cumulative)
        alpha[0] = alpha_cumulative[0]
        alpha[1:] = alpha_cumulative[1:] / alpha_cumulative[:-1]
        return torch.clip(alpha, 1e-4, 1.0 - 1e-8)

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
            "shift_schedule": self.shift_schedule,
            "img_size": self.img_size,
            "ref_img_size": self.ref_img_size,
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
        if self.shift_schedule:
            shift = (float(self.ref_img_size) / float(self.img_size)) ** 2
            print(f"Resolution shift: {self.ref_img_size} -> {self.img_size} "
                  f"(SNR x {shift:.5f})")
        print(f"Final alpha_bar: {self.alpha_cumulative[-1]:.6e}")
        print(f"Device: {self.device}")
        if self.schedule_kwargs:
            print(f"Schedule params: {self.schedule_kwargs}")
        print("=" * 50)
