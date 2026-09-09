"""Per-channel transform and standardization for multi-channel cell tensors.

Implements B11 of the v4 pipeline map:

    Transform, then standardize each channel; store the parameters.
    A low-variance channel is pure noise over most of the diffusion
    trajectory.  Standardization equalizes the signal-to-noise schedule
    across channels.  Sparse, heavy-tailed channels get log1p or a
    quantile map first.  The stored parameters are the decoder:
    unpacking a sample is just undoing this step.

The last sentence is the design constraint: the inverse has to be exact, not
approximate, because E1 unpacks every generated sample through it before the
descriptor ever runs.  So every transform here is a bijection on the reals with
a closed-form inverse, including outside the range the fit ever saw --- a
generated tensor is not bounded by the training data.

    forward(x)  = standardize(transform(x))
    inverse(z)  = transform_inverse(unstandardize(z))

Transforms
----------
``identity``
    For channels already on a comparable scale.
``log1p``
    ``log(1 + x)``, inverse ``expm1``.  For sparse, heavy-tailed counts;
    log1p keeps zero at zero, which matters when most units are empty.
``quantile``
    A monotone piecewise-linear map from the fitted marginal onto normal
    scores, extended by the end slopes outside the fitted range so it stays a
    bijection.  The inverse is the same piecewise-linear map with its knots
    swapped, which is exact.

Usage
-----
    pipeline = ChannelPipeline.fit(
        tensors,                              # (N, C, H, W) or iterable
        channel_names=["building", "access", "diversity"],
        transforms=["log1p", "identity", "identity"],
    )
    z = pipeline.forward(x)                   # into standardized space
    x_again = pipeline.inverse(z)             # exact round trip
    pipeline.save("channel_pipeline.json")
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

TRANSFORMS = ("identity", "log1p", "quantile")

# Number of probability levels the quantile map is fitted on.  257 knots put a
# knot every ~0.4 % of the distribution, which is finer than any marginal we
# read off a raster and cheap to store in the manifest.
DEFAULT_N_QUANTILES = 257

# The normal scores at the extremes are +-inf, so the fitted probability grid
# is clipped inward by this much at each end.
_P_EPS = 1e-4


# ---------------------------------------------------------------------------
# Piecewise-linear map with linear extrapolation
# ---------------------------------------------------------------------------


def _piecewise_linear(x: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Interpolate ``(xs, ys)`` at ``x``, extrapolating with the end slopes.

    ``xs`` must be strictly increasing with at least two entries.  Unlike
    ``np.interp``, which clamps, this stays a bijection on all of R --- the
    property the inverse depends on.
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.interp(x, xs, ys)

    left_slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
    right_slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])

    below = x < xs[0]
    above = x > xs[-1]
    out = np.where(below, ys[0] + (x - xs[0]) * left_slope, out)
    out = np.where(above, ys[-1] + (x - xs[-1]) * right_slope, out)
    return out


def _normal_ppf(p: np.ndarray) -> np.ndarray:
    """Inverse standard-normal CDF.  Imported locally to keep import cheap."""
    from scipy.special import ndtri

    return ndtri(p)


# ---------------------------------------------------------------------------
# One channel
# ---------------------------------------------------------------------------


@dataclass
class ChannelTransform:
    """Transform + standardization parameters for a single channel.

    Attributes
    ----------
    name:
        Channel name, matching the indicator specification table.
    transform:
        One of :data:`TRANSFORMS`.
    mean, std:
        Standardization applied *after* the transform.  ``std`` is never zero;
        a constant channel gets ``std = 1`` so the round trip still works.
    knots_x, knots_y:
        Quantile-map knots.  Empty for the other transforms.
    """

    name: str
    transform: str = "identity"
    mean: float = 0.0
    std: float = 1.0
    knots_x: list[float] = field(default_factory=list)
    knots_y: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.transform not in TRANSFORMS:
            raise ValueError(
                f"unknown transform {self.transform!r}; expected one of {TRANSFORMS}"
            )
        if self.transform == "quantile" and len(self.knots_x) < 2:
            raise ValueError(f"quantile transform for {self.name!r} needs >= 2 knots")

    # -- the transform, without standardization ----------------------------

    def _transform(self, x: np.ndarray) -> np.ndarray:
        if self.transform == "identity":
            return np.asarray(x, dtype=np.float64)
        if self.transform == "log1p":
            return np.log1p(np.asarray(x, dtype=np.float64))
        return _piecewise_linear(x, np.asarray(self.knots_x), np.asarray(self.knots_y))

    def _transform_inverse(self, y: np.ndarray) -> np.ndarray:
        if self.transform == "identity":
            return np.asarray(y, dtype=np.float64)
        if self.transform == "log1p":
            return np.expm1(np.asarray(y, dtype=np.float64))
        # The knots swapped: the exact functional inverse of a strictly
        # increasing piecewise-linear map.
        return _piecewise_linear(y, np.asarray(self.knots_y), np.asarray(self.knots_x))

    # -- the full encode / decode ------------------------------------------

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Data space -> standardized space."""
        return (self._transform(x) - self.mean) / self.std

    def inverse(self, z: np.ndarray) -> np.ndarray:
        """Standardized space -> data space.  Exact inverse of :meth:`forward`."""
        return self._transform_inverse(np.asarray(z, dtype=np.float64) * self.std + self.mean)

    # -- fitting ------------------------------------------------------------

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        name: str,
        transform: str = "identity",
        n_quantiles: int = DEFAULT_N_QUANTILES,
    ) -> "ChannelTransform":
        """Fit one channel's parameters to a flat array of its unit scores."""
        values = np.asarray(values, dtype=np.float64).ravel()
        if values.size == 0:
            raise ValueError(f"no values to fit channel {name!r}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"channel {name!r} contains non-finite values")

        knots_x: list[float] = []
        knots_y: list[float] = []

        if transform == "log1p" and values.min() <= -1.0:
            raise ValueError(
                f"channel {name!r} has values <= -1, so log1p is not invertible on it"
            )

        if transform == "quantile":
            knots_x, knots_y = _fit_quantile_knots(values, n_quantiles)
            transformed = _piecewise_linear(
                values, np.asarray(knots_x), np.asarray(knots_y)
            )
        elif transform == "log1p":
            transformed = np.log1p(values)
        else:
            transformed = values

        mean = float(transformed.mean())
        std = float(transformed.std())
        if std < 1e-12:
            # A constant channel carries no signal, but the round trip still
            # has to hold, so leave the scale alone rather than dividing by ~0.
            std = 1.0
        return cls(
            name=name,
            transform=transform,
            mean=mean,
            std=std,
            knots_x=knots_x,
            knots_y=knots_y,
        )

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "transform": self.transform,
            "mean": self.mean,
            "std": self.std,
        }
        if self.transform == "quantile":
            d["knots_x"] = self.knots_x
            d["knots_y"] = self.knots_y
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ChannelTransform":
        return cls(
            name=d["name"],
            transform=d.get("transform", "identity"),
            mean=float(d.get("mean", 0.0)),
            std=float(d.get("std", 1.0)),
            knots_x=list(d.get("knots_x", [])),
            knots_y=list(d.get("knots_y", [])),
        )


def _fit_quantile_knots(
    values: np.ndarray, n_quantiles: int
) -> tuple[list[float], list[float]]:
    """Knots mapping the empirical marginal onto normal scores.

    Duplicate quantile values --- which a sparse channel produces in bulk, since
    most of its mass sits on one number --- are collapsed to a single knot, so
    ``knots_x`` comes back strictly increasing and the map stays invertible.
    The collapsed knot takes the *mean* normal score of the run, placing the
    atom at the centre of the probability mass it owns rather than at one edge.

    Note what this cannot do.  A monotone map moves an atom, it does not spread
    one, so a channel where most units hold the same value keeps its spike
    whatever transform is chosen.  That is a property of the indicator, and
    it is what B10's marginals checkpoint is for; the transform only fixes the
    shape of the part of the channel that varies.
    """
    p = np.linspace(_P_EPS, 1.0 - _P_EPS, n_quantiles)
    xs = np.quantile(values, p)
    ys = _normal_ppf(p)

    # Collapse each run of equal xs to one knot at the run's mean normal score.
    boundaries = np.flatnonzero(np.r_[True, xs[1:] > xs[:-1]])
    run_ids = np.cumsum(np.r_[0, (xs[1:] > xs[:-1]).astype(int)])
    ys = np.bincount(run_ids, weights=ys) / np.bincount(run_ids)
    xs = xs[boundaries]

    if xs.size < 2:
        # Degenerate: every unit in the channel holds the same value.  Fall back
        # to a unit-slope map through that value so forward/inverse still work.
        centre = float(values[0])
        xs = np.array([centre - 0.5, centre + 0.5])
        ys = np.array([-0.5, 0.5])

    return [float(v) for v in xs], [float(v) for v in ys]


# ---------------------------------------------------------------------------
# All channels of a cell stack
# ---------------------------------------------------------------------------


class ChannelPipeline:
    """The stored decoder for a multi-channel cell dataset.

    Holds one :class:`ChannelTransform` per channel, in channel order, and
    applies them to ``(C, H, W)`` or ``(N, C, H, W)`` arrays.
    """

    def __init__(self, channels: Sequence[ChannelTransform]):
        if not channels:
            raise ValueError("a pipeline needs at least one channel")
        names = [c.name for c in channels]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate channel names: {names}")
        self.channels: list[ChannelTransform] = list(channels)

    def __len__(self) -> int:
        return len(self.channels)

    def __repr__(self) -> str:
        spec = ", ".join(f"{c.name}:{c.transform}" for c in self.channels)
        return f"ChannelPipeline({spec})"

    @property
    def channel_names(self) -> list[str]:
        return [c.name for c in self.channels]

    def index_of(self, name: str) -> int:
        return self.channel_names.index(name)

    def subset(self, names: Sequence[str]) -> "ChannelPipeline":
        """A pipeline over a subset of channels, in the order given."""
        by_name = {c.name: c for c in self.channels}
        missing = [n for n in names if n not in by_name]
        if missing:
            raise KeyError(f"no such channels: {missing}")
        return ChannelPipeline([by_name[n] for n in names])

    # -- fitting ------------------------------------------------------------

    @classmethod
    def fit(
        cls,
        tensors: Iterable[np.ndarray] | np.ndarray,
        channel_names: Sequence[str],
        transforms: Sequence[str] | str = "identity",
        n_quantiles: int = DEFAULT_N_QUANTILES,
    ) -> "ChannelPipeline":
        """Fit per-channel parameters over a collection of cell tensors.

        Parameters
        ----------
        tensors:
            An ``(N, C, H, W)`` array, or any iterable of ``(C, H, W)`` arrays.
            Iterables are consumed once and held in memory per channel, so pass
            a subsample for large datasets.
        channel_names:
            Names in channel order; its length fixes C.
        transforms:
            One name from :data:`TRANSFORMS` per channel, or a single name
            applied to every channel.
        """
        n_channels = len(channel_names)
        if isinstance(transforms, str):
            transforms = [transforms] * n_channels
        if len(transforms) != n_channels:
            raise ValueError(
                f"got {len(transforms)} transforms for {n_channels} channels"
            )

        per_channel = _stack_by_channel(tensors, n_channels)
        return cls(
            [
                ChannelTransform.fit(
                    per_channel[i], channel_names[i], transforms[i], n_quantiles
                )
                for i in range(n_channels)
            ]
        )

    # -- apply --------------------------------------------------------------

    def forward(self, x: np.ndarray, dtype=np.float64) -> np.ndarray:
        """Data space -> standardized space, for ``(C, H, W)`` or ``(N, C, H, W)``.

        Returns float64 by default.  Narrowing to float32 is the caller's
        choice, made where the precision is actually being traded for memory
        --- the loader does it on the way to the GPU --- because a float32
        round trip is no longer exact.
        """
        return self._apply(x, invert=False, dtype=dtype)

    def inverse(self, z: np.ndarray, dtype=np.float64) -> np.ndarray:
        """Standardized space -> data space.  Exact inverse of :meth:`forward`."""
        return self._apply(z, invert=True, dtype=dtype)

    def _apply(self, arr: np.ndarray, invert: bool, dtype=np.float64) -> np.ndarray:
        a = np.asarray(arr, dtype=np.float64)
        axis = self._channel_axis(a)
        out = np.empty_like(a)
        for i, channel in enumerate(self.channels):
            index = [slice(None)] * a.ndim
            index[axis] = i
            index_t = tuple(index)
            fn = channel.inverse if invert else channel.forward
            out[index_t] = fn(a[index_t])
        return out.astype(dtype, copy=False)

    def _channel_axis(self, a: np.ndarray) -> int:
        if a.ndim == 3:
            axis = 0
        elif a.ndim == 4:
            axis = 1
        else:
            raise ValueError(
                f"expected (C, H, W) or (N, C, H, W), got shape {a.shape}"
            )
        if a.shape[axis] != len(self.channels):
            raise ValueError(
                f"array has {a.shape[axis]} channels, pipeline has {len(self.channels)}"
            )
        return axis

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict:
        return {"channels": [c.to_dict() for c in self.channels]}

    @classmethod
    def from_dict(cls, d: dict) -> "ChannelPipeline":
        return cls([ChannelTransform.from_dict(c) for c in d["channels"]])

    def save(self, path: str) -> None:
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ChannelPipeline":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def _stack_by_channel(
    tensors: Iterable[np.ndarray] | np.ndarray, n_channels: int
) -> list[np.ndarray]:
    """Flatten a collection of ``(C, H, W)`` tensors into one array per channel."""
    if isinstance(tensors, np.ndarray) and tensors.ndim == 4:
        if tensors.shape[1] != n_channels:
            raise ValueError(
                f"tensors have {tensors.shape[1]} channels, expected {n_channels}"
            )
        return [tensors[:, i].ravel().astype(np.float64) for i in range(n_channels)]

    buckets: list[list[np.ndarray]] = [[] for _ in range(n_channels)]
    for tensor in tensors:
        t = np.asarray(tensor)
        if t.ndim != 3 or t.shape[0] != n_channels:
            raise ValueError(
                f"expected (C={n_channels}, H, W) tensors, got shape {t.shape}"
            )
        for i in range(n_channels):
            buckets[i].append(t[i].ravel().astype(np.float64))

    if not buckets[0]:
        raise ValueError("no tensors to fit on")
    return [np.concatenate(b) for b in buckets]
