"""Sampling methods for DDPM models."""

from sampling.ddim import sample_ddim
from sampling.ddpm_sample import sample_ddpm

__all__ = ["sample_ddim", "sample_ddpm"]
