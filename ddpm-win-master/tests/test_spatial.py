"""Tests for Moran's I and bivariate Moran's I (C2, E3c).

Synthetic grids with a known spatial structure, so the sign and rough size of
each statistic can be asserted rather than merely reproduced.

Run: pytest tests/test_spatial.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from analysis.spatial import (
    bivariate_morans_i,
    bivariate_morans_i_matrix,
    morans_i,
    spatial_lag,
)


def checkerboard(size=16):
    """Maximum dispersion: every unit's neighbours are its opposite."""
    y, x = np.indices((size, size))
    return ((y + x) % 2).astype(float)


def half_and_half(size=16):
    """Maximum clumping at this scale: one block high, one block low."""
    g = np.zeros((size, size))
    g[:, : size // 2] = 1.0
    return g


def noise(size=16, seed=0):
    return np.random.default_rng(seed).random((size, size))


def gradient(size=16):
    return np.tile(np.linspace(0.0, 1.0, size), (size, 1))


# ---------------------------------------------------------------------------
# Spatial lag
# ---------------------------------------------------------------------------


def test_lag_of_a_constant_grid_is_that_constant_including_at_the_edges():
    """Row-standardization at the boundary: average over the neighbours you have."""
    lag = spatial_lag(np.full((5, 5), 3.0))
    assert np.allclose(lag, 3.0)


def test_lag_averages_the_right_neighbours():
    grid = np.arange(9, dtype=float).reshape(3, 3)
    rook = spatial_lag(grid, connectivity=4)
    # Centre unit (value 4): neighbours 1, 3, 5, 7 -> mean 4.
    assert rook[1, 1] == pytest.approx(4.0)
    # Corner unit (value 0): neighbours 1 and 3 -> mean 2.
    assert rook[0, 0] == pytest.approx(2.0)

    queen = spatial_lag(grid, connectivity=8)
    # Centre unit: all eight others -> mean of 0..8 without 4 = 4.
    assert queen[1, 1] == pytest.approx(4.0)
    # Corner unit: neighbours 1, 3, 4 -> mean 8/3.
    assert queen[0, 0] == pytest.approx(8.0 / 3.0)


def test_lag_rejects_a_non_2d_grid():
    with pytest.raises(ValueError, match="H, W"):
        spatial_lag(np.zeros((3, 4, 4)))


def test_lag_rejects_an_unsupported_connectivity():
    with pytest.raises(ValueError, match="4 or 8"):
        spatial_lag(np.zeros((4, 4)), connectivity=6)


# ---------------------------------------------------------------------------
# Moran's I
# ---------------------------------------------------------------------------


def test_clumped_grid_is_near_one():
    assert morans_i(half_and_half()) > 0.85


def test_checkerboard_is_strongly_negative():
    assert morans_i(checkerboard(), connectivity=4) < -0.9


def test_noise_is_near_zero():
    assert abs(morans_i(noise())) < 0.15


def test_a_smooth_gradient_is_strongly_positive():
    assert morans_i(gradient()) > 0.9


def test_constant_grid_returns_zero_rather_than_nan():
    """The pilot's flattened building channel: no pattern, not a missing value."""
    assert morans_i(np.full((8, 8), 0.7)) == 0.0


def test_morans_i_is_invariant_to_shifting_and_scaling():
    grid = noise(seed=3)
    base = morans_i(grid)
    assert morans_i(grid * 100 + 7) == pytest.approx(base)


def test_morans_i_ordering_matches_intuition():
    rook = dict(connectivity=4)
    assert (
        morans_i(gradient(), **rook)
        > morans_i(noise(), **rook)
        > morans_i(checkerboard(), **rook)
    )


def test_queen_weights_read_a_checkerboard_as_almost_unstructured():
    """Not a defect: a checkerboard's diagonal neighbours are its own colour.

    Worth pinning, because at a 1.17 m unit the alley-and-rowhouse banding of
    HCMC is this kind of fine pattern, and the two connectivities give very
    different answers about it.
    """
    assert morans_i(checkerboard(), connectivity=4) < -0.9
    assert abs(morans_i(checkerboard(), connectivity=8)) < 0.1


# ---------------------------------------------------------------------------
# Bivariate Moran's I
# ---------------------------------------------------------------------------


def test_bivariate_with_itself_is_univariate_morans_i():
    grid = gradient()
    assert bivariate_morans_i(grid, grid) == pytest.approx(morans_i(grid))


def test_co_located_grids_score_high():
    """Buildings sitting on accessible units."""
    a = half_and_half()
    b = half_and_half() * 3.0 + 1.0
    assert bivariate_morans_i(a, b) > 0.85


def test_anti_located_grids_score_negative():
    a = half_and_half()
    b = 1.0 - half_and_half()
    assert bivariate_morans_i(a, b) < -0.85


def test_spatially_unrelated_grids_score_near_zero():
    assert abs(bivariate_morans_i(noise(seed=1), noise(seed=2))) < 0.15


def test_it_sees_layout_that_a_cell_average_correlation_cannot():
    """The E3c point, as a test.

    Two grids with the same per-cell mean and the same per-unit correlation of
    zero, but in one case each grid's mass sits beside the other's and in the
    other it does not.  A comparison of cell averages cannot tell them apart.
    """
    size = 16
    left = np.zeros((size, size))
    left[:, : size // 2] = 1.0
    right = np.zeros((size, size))
    right[:, size // 2 :] = 1.0

    scattered_a = np.zeros((size, size))
    scattered_a[::2, ::2] = 1.0
    scattered_b = np.zeros((size, size))
    scattered_b[1::2, 1::2] = 1.0

    # Same means on both sides, and both pairs are uncorrelated unit by unit.
    assert left.mean() == pytest.approx(right.mean())
    assert np.corrcoef(left.ravel(), right.ravel())[0, 1] < 0

    assert bivariate_morans_i(left, right) < bivariate_morans_i(
        scattered_a, scattered_b
    )


def test_bivariate_returns_zero_when_a_grid_is_constant():
    assert bivariate_morans_i(np.ones((8, 8)), noise(8)) == 0.0


def test_bivariate_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="shape"):
        bivariate_morans_i(np.zeros((8, 8)), np.zeros((8, 4)))


# ---------------------------------------------------------------------------
# The all-pairs matrix
# ---------------------------------------------------------------------------


def test_matrix_diagonal_is_each_channel_s_own_morans_i():
    cell = np.stack([gradient(), noise(seed=5), half_and_half()])
    matrix = bivariate_morans_i_matrix(cell)
    for i in range(3):
        assert matrix[i, i] == pytest.approx(morans_i(cell[i]))


def test_matrix_off_diagonal_matches_the_pairwise_function():
    cell = np.stack([gradient(), noise(seed=5), half_and_half()])
    matrix = bivariate_morans_i_matrix(cell)
    assert matrix[0, 2] == pytest.approx(bivariate_morans_i(cell[0], cell[2]))
    assert matrix[2, 0] == pytest.approx(bivariate_morans_i(cell[2], cell[0]))


def test_matrix_rejects_a_non_3d_tensor():
    with pytest.raises(ValueError, match="C, H, W"):
        bivariate_morans_i_matrix(np.zeros((8, 8)))
