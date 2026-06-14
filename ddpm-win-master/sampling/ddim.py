"""
DDIM (Denoising Diffusion Implicit Models) sampling for ddpm-win-master.

Ported from: Why-Diffusion-Models-Don-t-Memorize-main/Experiments/src/Utils/Diffusion.py
Reference: Song et al., "Denoising Diffusion Implicit Models" (arXiv:2010.02502)

Adapted for ddpm-win-master's DDPM model interface, DiffusionConfig schedule tensors,
and TrainingConfig parameter system.
"""

import numpy as np
import torch
from typing import Optional, List, Tuple, Union


def _gather_4d(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather values from a 1D schedule tensor at timestep indices, reshape for 4D broadcast.

    Equivalent to the research repo's ``get(element, t, dim=4)`` helper.

    Args:
        tensor: 1D tensor of shape (T,) with schedule values.
        indices: 1D tensor of shape (B,) with timestep indices.

    Returns:
        Tensor of shape (B, 1, 1, 1) for broadcasting with (B, C, H, W) images.
    """
    return tensor.to(indices.device)[indices][:, None, None, None]


@torch.no_grad()
def sample_ddim(
    model,
    n_images: int = 25,
    config=None,
    diffusion_config=None,
    device: str = "cuda",
    eta: float = 0.0,
    ddim_steps: int = 50,
    return_trajectory: bool = False,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]],
]:
    """Generate images using DDIM sampling.

    Ported from ``sample_diffusion_from_noise_DDIM`` in the research repo and
    adapted for ddpm-win-master's model / config interfaces.

    Args:
        model: DDPM model instance (from ``model.ddpm.DDPM``). Must expose
            ``model.eps_model(x, t)`` for noise prediction.
        n_images: Number of images to generate.
        config: ``TrainingConfig`` with ``timesteps``, ``img_channels``,
            ``img_size``. Falls back to model attributes and sensible defaults
            if *None*.
        diffusion_config: ``DiffusionConfig`` (from ``model.diffusion_config``)
            with pre-computed schedule tensors. If *None*, derived from
            ``model.alpha_bar``.
        device: Torch device string (``'cuda'``, ``'cpu'``, etc.).
        eta: Stochasticity parameter. ``eta=0`` gives deterministic DDIM;
            ``eta=1`` recovers DDPM-level stochasticity.
        ddim_steps: Number of subsampled DDIM steps. Fewer steps = faster
            sampling at the cost of quality.
        return_trajectory: If *True*, also return a list of intermediate
            samples (one per DDIM step) for analysis.

    Returns:
        ``(generated_images, initial_noise)`` if *return_trajectory* is *False*.
        ``(generated_images, initial_noise, trajectory)`` otherwise, where
        *trajectory* is a list of ``ddim_steps`` tensors each of shape
        ``(n_images, C, H, W)``.
    """
    # ------------------------------------------------------------------
    # Resolve config defaults
    # ------------------------------------------------------------------
    if config is not None:
        timesteps = config.timesteps
        img_channels = config.img_channels
        img_size = config.img_size
    else:
        timesteps = model.timesteps
        img_channels = 1
        img_size = 32

    # ------------------------------------------------------------------
    # Resolve schedule tensors
    # ------------------------------------------------------------------
    if diffusion_config is not None:
        sqrt_alpha_cum = diffusion_config.sqrt_alpha_cumulative.to(device)
        sqrt_one_minus_alpha_cum = diffusion_config.sqrt_one_minus_alpha_cumulative.to(
            device
        )
    else:
        # Derive from model's registered buffers
        alpha_bar = model.alpha_bar.to(device)
        sqrt_alpha_cum = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_cum = torch.sqrt(1.0 - alpha_bar)

    total_steps = timesteps
    if ddim_steps is None or ddim_steps <= 0:
        ddim_steps = total_steps

    # Create subsampled timestep schedule (high → low noise)
    time_steps = np.linspace(0, total_steps - 1, ddim_steps, dtype=int)[::-1]
    time_steps = list(time_steps)

    # ------------------------------------------------------------------
    # Generate initial noise
    # ------------------------------------------------------------------
    x = torch.randn(n_images, img_channels, img_size, img_size, device=device)
    x_init = x.clone()

    trajectory: Optional[List[torch.Tensor]] = []
    if return_trajectory:
        trajectory.append(x.clone())

    # ------------------------------------------------------------------
    # Resolve noise-prediction callable
    # ------------------------------------------------------------------
    # Accept both a full DDPM wrapper and a bare eps_model (e.g. U-Net)
    if hasattr(model, "eps_model"):
        eps_fn = model.eps_model
    else:
        eps_fn = model

    # ------------------------------------------------------------------
    # DDIM reverse diffusion loop
    # ------------------------------------------------------------------
    eps_fn.eval()
    for i in range(len(time_steps) - 1):
        t = time_steps[i]
        t_prev = time_steps[i + 1]

        ts = torch.full((n_images,), t, dtype=torch.long, device=device)

        # Noise prediction
        eps_ts = eps_fn(x, ts)

        # Scaling factors for current timestep t
        sqrt_one_minus_alpha_cum_t = _gather_4d(sqrt_one_minus_alpha_cum, ts)
        sqrt_alpha_cum_t = _gather_4d(sqrt_alpha_cum, ts)

        # Estimate the clean image x0
        x0_pred = (x - sqrt_one_minus_alpha_cum_t * eps_ts) / sqrt_alpha_cum_t

        # Scaling factors for previous timestep t_prev
        # t_prev is a scalar (same for all samples), reshape for broadcast
        sqrt_one_minus_alpha_cum_t_prev = sqrt_one_minus_alpha_cum[t_prev].reshape(
            1, 1, 1, 1
        )
        sqrt_alpha_cum_t_prev = sqrt_alpha_cum[t_prev].reshape(1, 1, 1, 1)

        # Compute sigma_t (stochastic component)
        sigma_t = eta * torch.sqrt(
            (sqrt_one_minus_alpha_cum_t_prev**2 / (sqrt_one_minus_alpha_cum_t**2))
            * (1.0 - (sqrt_alpha_cum_t**2) / (sqrt_alpha_cum_t_prev**2))
        )

        # Stochastic noise (zero at the final step)
        z = torch.randn_like(x) if t_prev > 0 else torch.zeros_like(x)

        # DDIM update (Eq. 12 from arXiv:2010.02502)
        x = (
            sqrt_alpha_cum_t_prev * x0_pred
            + torch.sqrt(
                torch.clamp(sqrt_one_minus_alpha_cum_t_prev**2 - sigma_t**2, min=1e-8)
            )
            * eps_ts
            + sigma_t * z
        )

        if return_trajectory:
            trajectory.append(x.clone())

    if return_trajectory:
        return x, x_init, trajectory
    return x, x_init
