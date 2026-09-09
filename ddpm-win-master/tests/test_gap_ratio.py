"""Tests for the k-NN gap ratio and F_mem on standardized tensors (E2, D2).

The metric has to give the right answer on cases where the answer is known by
construction: a generated cell that *is* a training cell, and one that sits
nowhere near any of them.  And it has to refuse unstandardized input, which is
the mistake that made the pilot's F_mem a statement about accessibility alone.

Run: pytest tests/test_gap_ratio.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch

from evaluation.gap_ratio import (
    DEFAULT_GAP_THRESHOLD,
    check_standardized,
    compute_fmem,
    knn_gap_ratio,
    memorization_fraction,
)


def standardized_cells(n=32, c=3, size=8, seed=0):
    """Cells that pass the standardization check by construction."""
    rng = np.random.default_rng(seed)
    cells = rng.normal(0.0, 1.0, (n, c, size, size))
    return (cells - cells.mean(axis=(0, 2, 3), keepdims=True)) / cells.std(
        axis=(0, 2, 3), keepdims=True
    )


# ---------------------------------------------------------------------------
# The gap ratio itself
# ---------------------------------------------------------------------------


def test_an_exact_copy_scores_essentially_zero():
    """Not exactly zero: distances are accumulated in float32.

    An identical cell comes back at ~1e-4 rather than 0, which is three orders
    of magnitude below the 1/3 threshold and so cannot change F_mem. Worth
    pinning, because at 1024 px the accumulation is over six million terms and
    the residual grows with the dimension.
    """
    train = standardized_cells()
    ratios = knn_gap_ratio(train, train[[3, 7]], check=False)
    assert np.all(ratios < 1e-2)


def test_a_near_copy_scores_below_the_threshold():
    train = standardized_cells()
    nudged = train[[5]] + 1e-3
    assert knn_gap_ratio(train, nudged, check=False)[0] < DEFAULT_GAP_THRESHOLD


def test_a_cell_far_from_everything_scores_near_one():
    """Its two nearest neighbours are then equally far, which is the point."""
    train = standardized_cells()
    far = np.full((1, 3, 8, 8), 500.0)
    assert knn_gap_ratio(train, far, check=False)[0] > 0.99


def test_ratios_stay_in_the_unit_interval():
    train = standardized_cells()
    ratios = knn_gap_ratio(train, standardized_cells(n=16, seed=1), check=False)
    assert np.all(ratios >= 0.0) and np.all(ratios <= 1.0)


def test_the_ratio_is_d1_over_d2_and_nothing_else():
    """Checked against a direct computation on a tiny set."""
    train = np.zeros((3, 1, 2, 2))
    train[0] = 0.0
    train[1] = 1.0
    train[2] = 4.0
    generated = np.full((1, 1, 2, 2), 0.5)

    # Flattened distances: 1.0, 1.0, 7.0.  d1 / d2 = 1.
    got = knn_gap_ratio(train, generated, check=False)[0]
    assert got == pytest.approx(1.0)


def test_batching_does_not_change_the_answer():
    train = standardized_cells()
    generated = standardized_cells(n=17, seed=2)
    one_batch = knn_gap_ratio(train, generated, batch_size=1000, check=False)
    many = knn_gap_ratio(train, generated, batch_size=3, check=False)
    assert np.allclose(one_batch, many, atol=1e-6)


def test_it_accepts_torch_tensors_and_numpy_alike():
    train = standardized_cells()
    generated = standardized_cells(n=4, seed=3)
    from_numpy = knn_gap_ratio(train, generated, check=False)
    from_torch = knn_gap_ratio(
        torch.from_numpy(train), torch.from_numpy(generated), check=False
    )
    assert np.allclose(from_numpy, from_torch, atol=1e-5)


def test_duplicate_training_cells_do_not_read_as_model_memorization():
    """B1 forbids overlapping windows; if one slips through, d2 is 0."""
    train = standardized_cells(n=4)
    train[1] = train[0]
    ratio = knn_gap_ratio(train, train[[0]], check=False)[0]
    assert ratio == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# The standardization guard
# ---------------------------------------------------------------------------


def test_the_guard_passes_standardized_cells():
    check_standardized(standardized_cells())


def test_the_guard_catches_a_channel_with_a_runaway_scale():
    """The pilot's failure: one channel decides every distance."""
    cells = standardized_cells()
    cells[:, 1] *= 50.0
    with pytest.raises(ValueError, match="do not look standardized"):
        check_standardized(cells)


def test_the_error_names_the_offending_channel_and_its_std():
    cells = standardized_cells()
    cells[:, 2] *= 30.0
    with pytest.raises(ValueError, match=r"\{2:"):
        check_standardized(cells)


def test_knn_gap_ratio_checks_both_sides_by_default():
    train = standardized_cells()
    unstandardized = standardized_cells(n=8, seed=4) * 40.0
    with pytest.raises(ValueError, match="generated_tensors"):
        knn_gap_ratio(train, unstandardized)


def test_the_check_can_be_turned_off_deliberately():
    train = standardized_cells()
    ratios = knn_gap_ratio(train, standardized_cells(n=4, seed=5) * 40.0, check=False)
    assert ratios.shape == (4,)


def test_standardized_and_unstandardized_distances_disagree_about_which_cell_is_nearest():
    """Why E2 insists on the standardized space, made concrete.

    Two channels, one with a hundredfold wider range.  In data space the wide
    channel decides the nearest neighbour; standardized, the narrow one can.
    """
    train = np.zeros((2, 2, 4, 4))
    train[0] = [np.zeros((4, 4)), np.zeros((4, 4))]        # narrow 0,  wide 0
    train[1] = [np.full((4, 4), 10.0), np.ones((4, 4))]    # narrow 10, wide 1
    generated = np.zeros((1, 2, 4, 4))
    generated[0, 0] = 1.0                                  # close to train 0
    generated[0, 1] = 0.9                                  # close to train 1

    scale = np.array([1.0, 100.0]).reshape(1, 2, 1, 1)
    data_space = train * scale
    query_data_space = generated * scale

    def nearest(train_set, query):
        flat_train = train_set.reshape(len(train_set), -1)
        return int(
            np.argmin(((flat_train - query.reshape(1, -1)) ** 2).sum(axis=1))
        )

    assert nearest(train, generated) == 0
    assert nearest(data_space, query_data_space) == 1


# ---------------------------------------------------------------------------
# F_mem
# ---------------------------------------------------------------------------


def test_fmem_is_one_when_every_sample_is_a_copy():
    train = standardized_cells()
    result = compute_fmem(train, train[:8], n_bootstrap=100, check=False)
    assert result["fraction_mem"] == pytest.approx(1.0)
    assert result["n_memorized"] == 8
    assert result["n_total"] == 8


def test_fmem_is_zero_when_nothing_is_a_copy():
    train = standardized_cells()
    far = np.full((8, 3, 8, 8), 500.0) + np.arange(8).reshape(8, 1, 1, 1)
    result = compute_fmem(train, far, n_bootstrap=100, check=False)
    assert result["fraction_mem"] == pytest.approx(0.0)


def test_fmem_reports_a_confidence_interval_around_its_estimate():
    ratios = np.concatenate([np.full(30, 0.1), np.full(70, 0.9)])
    result = memorization_fraction(ratios, n_bootstrap=500, random_state=0)
    assert result["fraction_mem"] == pytest.approx(0.3, abs=0.05)
    assert result["lower_ci"] < result["fraction_mem"] < result["upper_ci"]
    assert result["n_memorized"] == 30


def test_the_threshold_is_disclosed_in_the_result():
    """E4: "Disclose thresholds"."""
    result = memorization_fraction(np.full(10, 0.5), n_bootstrap=50)
    assert result["gap_threshold"] == DEFAULT_GAP_THRESHOLD


def test_a_different_threshold_moves_the_answer():
    ratios = np.full(100, 0.5)
    assert memorization_fraction(ratios, 1 / 3, n_bootstrap=50)["fraction_mem"] == 0.0
    assert memorization_fraction(ratios, 0.9, n_bootstrap=50)["fraction_mem"] == 1.0


def test_bootstrapping_is_reproducible_under_a_seed():
    ratios = np.random.default_rng(0).random(200)
    a = memorization_fraction(ratios, n_bootstrap=200, random_state=1)
    b = memorization_fraction(ratios, n_bootstrap=200, random_state=1)
    assert a["fraction_mem"] == b["fraction_mem"]
    assert a["lower_ci"] == b["lower_ci"]


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_a_single_training_cell_has_no_second_nearest_neighbour():
    with pytest.raises(ValueError, match="second-nearest"):
        knn_gap_ratio(standardized_cells(n=1), standardized_cells(n=2), check=False)


def test_mismatched_cell_shapes_are_refused():
    with pytest.raises(ValueError, match="generated are"):
        knn_gap_ratio(
            standardized_cells(c=3), standardized_cells(c=6), check=False
        )


def test_unbatched_tensors_are_refused():
    with pytest.raises(ValueError, match=r"N, C, H, W"):
        knn_gap_ratio(standardized_cells(), standardized_cells()[0], check=False)


def test_an_empty_set_of_ratios_is_refused():
    with pytest.raises(ValueError, match="no gap ratios"):
        memorization_fraction(np.array([]))
