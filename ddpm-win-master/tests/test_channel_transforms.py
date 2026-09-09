"""Tests for the per-channel transform and standardization (B11).

The property that matters is the one E1 depends on: ``inverse`` undoes
``forward`` exactly, in physical units, including on values no cell in the
fitting set ever held --- a generated sample is not bounded by the training
data.

Run: pytest tests/test_channel_transforms.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from datasets.channel_transforms import (
    ChannelPipeline,
    ChannelTransform,
    _piecewise_linear,
)

ROUND_TRIP_TOL = 1e-9


def synthetic_cells(n=64, size=16, seed=0):
    """Four channels with the shapes the v4 channels are expected to have.

    building_density  bounded [0, 1], mildly skewed
    accessibility     smooth, strictly positive, log-normal-ish
    poi_count         sparse counts, ~90 % zeros, heavy tail
    poi_entropy       bounded [0, log k], zero-inflated
    """
    rng = np.random.default_rng(seed)
    building = rng.beta(0.6, 3.0, (n, size, size))
    access = rng.lognormal(0.0, 1.0, (n, size, size))
    sparse_mask = rng.random((n, size, size)) < 0.10
    poi_count = np.where(sparse_mask, rng.poisson(4.0, (n, size, size)), 0).astype(float)
    entropy = np.where(sparse_mask, rng.uniform(0.0, np.log(8), (n, size, size)), 0.0)
    return np.stack([building, access, poi_count, entropy], axis=1)


CHANNEL_NAMES = ["building_density", "accessibility", "poi_count", "poi_entropy"]


# ---------------------------------------------------------------------------
# The piecewise-linear helper
# ---------------------------------------------------------------------------


def test_piecewise_linear_extrapolates_instead_of_clamping():
    xs = np.array([0.0, 1.0, 2.0])
    ys = np.array([0.0, 10.0, 20.0])
    got = _piecewise_linear(np.array([-1.0, 0.5, 3.0]), xs, ys)
    # np.interp would give [0, 5, 20]; extrapolation keeps the map a bijection.
    assert np.allclose(got, [-10.0, 5.0, 30.0])


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("transform", ["identity", "log1p", "quantile"])
def test_round_trip_is_exact_on_fitted_data(transform):
    cells = synthetic_cells()
    pipeline = ChannelPipeline.fit(cells, CHANNEL_NAMES, transform)
    z = pipeline.forward(cells)
    back = pipeline.inverse(z)
    assert np.abs(back - cells).max() < ROUND_TRIP_TOL


def test_round_trip_with_a_different_transform_per_channel():
    cells = synthetic_cells()
    pipeline = ChannelPipeline.fit(
        cells, CHANNEL_NAMES, ["identity", "log1p", "quantile", "quantile"]
    )
    back = pipeline.inverse(pipeline.forward(cells))
    assert np.abs(back - cells).max() < ROUND_TRIP_TOL


@pytest.mark.parametrize("transform", ["identity", "log1p", "quantile"])
def test_round_trip_holds_outside_the_fitted_range(transform):
    """A generated sample lands wherever it lands; the decoder must still work."""
    cells = synthetic_cells()
    pipeline = ChannelPipeline.fit(cells, CHANNEL_NAMES, transform)

    rng = np.random.default_rng(1)
    # Standardized values far outside anything the fit saw.
    z = rng.normal(0.0, 6.0, (8, 4, 16, 16))
    x = pipeline.inverse(z)
    assert np.all(np.isfinite(x))
    z_again = pipeline.forward(x)
    assert np.abs(z_again - z).max() < 1e-6


def test_inverse_is_the_inverse_not_an_approximation_of_the_identity():
    """Guards against a transform whose inverse is only right near the mean."""
    cells = synthetic_cells()
    pipeline = ChannelPipeline.fit(cells, CHANNEL_NAMES, "quantile")
    tail = cells.max(axis=(0,), keepdims=True) * 3.0  # beyond every fitted value
    back = pipeline.inverse(pipeline.forward(tail))
    assert np.abs(back - tail).max() < 1e-8


# ---------------------------------------------------------------------------
# Standardization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("transform", ["identity", "log1p", "quantile"])
def test_standardized_channels_have_zero_mean_and_unit_variance(transform):
    """B11's stated purpose: equalize the SNR schedule across channels."""
    cells = synthetic_cells()
    z = ChannelPipeline.fit(cells, CHANNEL_NAMES, transform).forward(cells)
    assert np.abs(z.mean(axis=(0, 2, 3))).max() < 1e-8
    assert np.abs(z.std(axis=(0, 2, 3)) - 1.0).max() < 1e-8


def test_quantile_transform_gaussianizes_a_continuous_heavy_tailed_channel():
    """Why the option exists: accessibility is smooth and log-normal-ish."""
    from scipy.stats import kurtosis

    access = synthetic_cells()[:, 1]
    identity_z = ChannelTransform.fit(access, "accessibility", "identity").forward(access)
    quantile_z = ChannelTransform.fit(access, "accessibility", "quantile").forward(access)

    assert abs(kurtosis(quantile_z.ravel())) < abs(kurtosis(identity_z.ravel()))
    assert abs(kurtosis(quantile_z.ravel())) < 0.1


def test_no_transform_can_spread_a_zero_inflated_channel():
    """A limit worth having in the test suite rather than discovering in B11.

    A monotone map moves an atom, it does not spread one.  A channel where 90 %
    of units hold exactly zero keeps its spike under identity, log1p and the
    quantile map alike, so the transform is not the lever for sparsity ---
    B10's marginals and the indicator's own ring scale are.
    """
    from scipy.stats import kurtosis

    poi = synthetic_cells()[:, 2]
    kurtoses = {
        name: abs(kurtosis(ChannelTransform.fit(poi, "poi_count", name).forward(poi).ravel()))
        for name in ("identity", "log1p", "quantile")
    }
    assert min(kurtoses.values()) > 5.0, kurtoses


def test_constant_channel_survives_the_round_trip():
    """std would be 0; the fit must not divide by it."""
    cells = synthetic_cells()
    cells[:, 0] = 0.25
    pipeline = ChannelPipeline.fit(cells, CHANNEL_NAMES, "identity")
    back = pipeline.inverse(pipeline.forward(cells))
    assert np.abs(back - cells).max() < ROUND_TRIP_TOL


def test_all_zero_sparse_channel_survives_the_quantile_fit():
    """The degenerate end of "expected sparsity": a channel with no signal."""
    cells = synthetic_cells()
    cells[:, 2] = 0.0
    pipeline = ChannelPipeline.fit(cells, CHANNEL_NAMES, "quantile")
    back = pipeline.inverse(pipeline.forward(cells))
    assert np.abs(back - cells).max() < ROUND_TRIP_TOL


# ---------------------------------------------------------------------------
# Shapes, subsets, persistence
# ---------------------------------------------------------------------------


def test_accepts_single_cells_and_batches():
    cells = synthetic_cells()
    pipeline = ChannelPipeline.fit(cells, CHANNEL_NAMES, "log1p")
    one = pipeline.forward(cells[0])
    assert one.shape == cells[0].shape
    assert np.allclose(one, pipeline.forward(cells)[0])


def test_rejects_a_channel_count_mismatch():
    cells = synthetic_cells()
    pipeline = ChannelPipeline.fit(cells, CHANNEL_NAMES, "identity")
    with pytest.raises(ValueError, match="channels"):
        pipeline.forward(cells[:, :2])


def test_fit_accepts_an_iterable_of_cells():
    cells = synthetic_cells()
    from_array = ChannelPipeline.fit(cells, CHANNEL_NAMES, "log1p")
    from_iter = ChannelPipeline.fit(iter(cells), CHANNEL_NAMES, "log1p")
    for a, b in zip(from_array.channels, from_iter.channels):
        assert a.mean == pytest.approx(b.mean)
        assert a.std == pytest.approx(b.std)


def test_subset_keeps_each_channel_s_own_parameters():
    cells = synthetic_cells()
    pipeline = ChannelPipeline.fit(cells, CHANNEL_NAMES, "quantile")
    subset = pipeline.subset(["poi_count", "building_density"])

    assert subset.channel_names == ["poi_count", "building_density"]
    reordered = cells[:, [2, 0]]
    assert np.allclose(subset.forward(reordered), pipeline.forward(cells)[:, [2, 0]])


def test_save_and_load_reproduce_the_decoder_exactly(tmp_path):
    cells = synthetic_cells()
    pipeline = ChannelPipeline.fit(
        cells, CHANNEL_NAMES, ["identity", "log1p", "quantile", "quantile"]
    )
    path = str(tmp_path / "pipeline.json")
    pipeline.save(path)
    loaded = ChannelPipeline.load(path)

    assert loaded.channel_names == pipeline.channel_names
    assert np.abs(loaded.forward(cells) - pipeline.forward(cells)).max() < 1e-12
    assert np.abs(loaded.inverse(pipeline.forward(cells)) - cells).max() < ROUND_TRIP_TOL


def test_log1p_fit_refuses_values_it_cannot_invert():
    cells = synthetic_cells()
    cells[:, 1] = -2.0
    with pytest.raises(ValueError, match="log1p"):
        ChannelPipeline.fit(cells, CHANNEL_NAMES, "log1p")


def test_duplicate_channel_names_are_rejected():
    cells = synthetic_cells()
    with pytest.raises(ValueError, match="duplicate"):
        ChannelPipeline.fit(cells, ["a", "b", "a", "c"], "identity")
