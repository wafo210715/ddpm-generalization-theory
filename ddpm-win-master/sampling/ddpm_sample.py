"""
DDPM (Denoising Diffusion Probabilistic Models) sampling for ddpm-win-master.

Reference: Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
Algorithm 2 — full 1000-step stochastic reverse diffusion.

Adapted for ddpm-win-master's DDPM model interface, DiffusionConfig schedule tensors,
and TrainingConfig parameter system.  Matches the ``sample_ddim()`` interface in
``sampling/ddim.py`` so both can be used interchangeably.
"""

from typing import Optional, List, Tuple, Union

import torch
from tqdm.auto import tqdm

from sampling.ddim import _gather_4d


@torch.no_grad()
def sample_ddpm(
    model,
    n_images: int = 25,
    config=None,
    diffusion_config=None,
    device: str = "cuda",
    return_trajectory: bool = False,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]],
]:
    """Generate images using the full DDPM reverse process (Algorithm 2).

    Args:
        model: DDPM model instance (from ``model.ddpm.DDPM``). Must expose
            ``model.eps_model(x, t)`` for noise prediction.
        n_images: Number of images to generate.
        config: ``TrainingConfig`` with ``timesteps``, ``img_channels``,
            ``img_size``. Falls back to model attributes and sensible defaults
            if *None*.
        diffusion_config: ``DiffusionConfig`` (from ``model.diffusion_config``)
            with pre-computed schedule tensors. If *None*, derived from
            ``model.alpha``, ``model.alpha_bar``, ``model.beta``.
        device: Torch device string (``'cuda'``, ``'cpu'``, etc.).
        return_trajectory: If *True*, also return a list of intermediate
            samples (one per timestep) for analysis.

    Returns:
        ``(generated_images, initial_noise)`` if *return_trajectory* is *False*.
        ``(generated_images, initial_noise, trajectory)`` otherwise, where
        *trajectory* is a list of ``timesteps`` tensors each of shape
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
        alpha = diffusion_config.alpha.to(device)
        alpha_bar = diffusion_config.alpha_cumulative.to(device)
        beta = diffusion_config.beta.to(device)
        sqrt_one_minus_alpha_cum = diffusion_config.sqrt_one_minus_alpha_cumulative.to(
            device
        )
    else:
        alpha = model.alpha.to(device)
        alpha_bar = model.alpha_bar.to(device)
        beta = model.beta.to(device)
        sqrt_one_minus_alpha_cum = torch.sqrt(1.0 - alpha_bar)

    # ------------------------------------------------------------------
    # Generate initial noise  x_T ~ N(0, I)
    # ------------------------------------------------------------------
    x = torch.randn(n_images, img_channels, img_size, img_size, device=device)
    x_init = x.clone()

    trajectory: Optional[List[torch.Tensor]] = []
    if return_trajectory:
        trajectory.append(x.clone())

    # ------------------------------------------------------------------
    # Resolve noise-prediction callable
    # ------------------------------------------------------------------
    if hasattr(model, "eps_model"):
        eps_fn = model.eps_model
    else:
        eps_fn = model

    eps_fn.eval()

    # ------------------------------------------------------------------
    # DDPM reverse diffusion  (Algorithm 2, Ho et al. 2020)
    # ------------------------------------------------------------------
    # x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * eps_theta(x_t, t))
    #           + sigma_t * z
    # where sigma_t = sqrt(beta_t) for t > 0, sigma_t = 0 for t = 0
    # ------------------------------------------------------------------
    for t in tqdm(range(timesteps - 1, -1, -1), desc="DDPM sampling", total=timesteps):
        ts = torch.full((n_images,), t, dtype=torch.long, device=device)

        # Predict noise
        eps_theta = eps_fn(x, ts)

        # Gather per-sample schedule values  →  (B, 1, 1, 1)
        alpha_t = _gather_4d(alpha, ts)
        sqrt_one_minus_alpha_bar_t = _gather_4d(sqrt_one_minus_alpha_cum, ts)
        beta_t = _gather_4d(beta, ts)

        # Reverse mean
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x - (beta_t / sqrt_one_minus_alpha_bar_t) * eps_theta
        )

        # Stochastic noise
        if t > 0:
            z = torch.randn_like(x)
            x = mean + torch.sqrt(beta_t) * z
        else:
            x = mean

        if return_trajectory:
            trajectory.append(x.clone())

    x = x.clamp(-1, 1)

    if return_trajectory:
        return x, x_init, trajectory
    return x, x_init
