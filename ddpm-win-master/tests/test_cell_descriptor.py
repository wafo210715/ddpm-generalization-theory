"""Tests for the cell descriptor (C2).

The plan's constraints, as tests: one function for real and generated cells,
computable from the raster alone, and stable in its keys so two sets of
descriptors are actually comparable.

Run: pytest tests/test_cell_descriptor.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from analysis.cell_descriptor import (
    CHANNEL_STATISTICS,
    describe_cell,
    describe_cells,
    descriptor_keys,
)
from analysis.spatial import morans_i

NAMES = ["building", "accessibility", "poi_count"]


def synthetic_cell(size=16, seed=0):
    rng = np.random.default_rng(seed)
    building = (rng.random((size, size)) < 0.3).astype(float)
    y, x = np.indices((size, size))
    accessibility = np.exp(-((y - size / 2) ** 2 + (x - size / 2) ** 2) / (2 * 25))
    poi_count = np.where(rng.random((size, size)) < 0.08, rng.poisson(3, (size, size)), 0.0)
    return np.stack([building, accessibility, poi_count])


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------


def test_every_channel_gets_every_statistic():
    d = describe_cell(synthetic_cell(), NAMES)
    for name in NAMES:
        for stat in CHANNEL_STATISTICS:
            assert f"{name}_{stat}" in d


def test_cross_channel_correlations_cover_each_unordered_pair():
    d = describe_cell(synthetic_cell(), NAMES)
    assert "corr_building_accessibility" in d
    assert "corr_building_poi_count" in d
    assert "corr_accessibility_poi_count" in d
    assert "corr_accessibility_building" not in d  # not both orders


def test_descriptor_keys_predicts_the_keys_without_computing_them():
    assert set(descriptor_keys(NAMES)) == set(describe_cell(synthetic_cell(), NAMES))


def test_the_descriptor_is_flat_and_all_numeric():
    d = describe_cell(synthetic_cell(), NAMES)
    assert all(isinstance(v, float) for v in d.values())
    assert all(np.isfinite(v) for v in d.values())


@pytest.mark.parametrize("n_channels", [1, 3, 6, 14])
def test_it_works_for_any_channel_count(n_channels):
    rng = np.random.default_rng(0)
    cell = rng.random((n_channels, 8, 8))
    names = [f"c{i}" for i in range(n_channels)]
    d = describe_cell(cell, names)
    assert len(d) == len(descriptor_keys(names))


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------


def test_the_statistics_are_the_ones_they_are_named_after():
    cell = synthetic_cell()
    d = describe_cell(cell, NAMES)
    assert d["building_mean"] == pytest.approx(cell[0].mean())
    assert d["accessibility_std"] == pytest.approx(cell[1].std())
    assert d["building_morans_i"] == pytest.approx(morans_i(cell[0]))


def test_zero_fraction_counts_empty_units():
    cell = np.zeros((3, 8, 8))
    cell[0, :4] = 1.0
    d = describe_cell(cell, NAMES)
    assert d["building_zero_fraction"] == pytest.approx(0.5)
    assert d["accessibility_zero_fraction"] == pytest.approx(1.0)


def test_zero_atol_lets_a_smooth_field_count_near_zero_as_empty():
    cell = np.full((3, 8, 8), 1e-9)
    assert describe_cell(cell, NAMES)["building_zero_fraction"] == 0.0
    assert describe_cell(cell, NAMES, zero_atol=1e-6)["building_zero_fraction"] == 1.0


def test_correlation_matches_numpy():
    cell = synthetic_cell()
    d = describe_cell(cell, NAMES)
    expected = np.corrcoef(cell[0].ravel(), cell[1].ravel())[0, 1]
    assert d["corr_building_accessibility"] == pytest.approx(expected)


def test_a_constant_channel_gives_zeros_not_nans():
    """A cell with no buildings at all is a real cell, not an error."""
    cell = synthetic_cell()
    cell[0] = 0.0
    d = describe_cell(cell, NAMES)
    assert d["building_std"] == 0.0
    assert d["building_morans_i"] == 0.0
    assert d["building_skewness"] == 0.0
    assert d["corr_building_accessibility"] == 0.0
    assert all(np.isfinite(v) for v in d.values())


def test_skewness_has_the_expected_sign():
    size = 16
    right_skewed = np.zeros((3, size, size))
    right_skewed[0] = np.concatenate(
        [np.zeros(size * size - 8), np.full(8, 10.0)]
    ).reshape(size, size)
    assert describe_cell(right_skewed, NAMES)["building_skewness"] > 1.0


# ---------------------------------------------------------------------------
# One function for both sides
# ---------------------------------------------------------------------------


def test_the_same_function_describes_a_generated_cell():
    """C2: generated cells have no vectors behind them, only a raster."""
    rng = np.random.default_rng(7)
    generated = rng.normal(0.5, 0.3, (3, 16, 16))
    real = describe_cell(synthetic_cell(), NAMES)
    fake = describe_cell(generated, NAMES)
    assert set(real) == set(fake)


def test_it_refuses_a_cell_with_non_finite_values():
    cell = synthetic_cell()
    cell[1, 0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        describe_cell(cell, NAMES)


def test_it_refuses_a_name_count_mismatch():
    with pytest.raises(ValueError, match="channel names"):
        describe_cell(synthetic_cell(), ["a", "b"])


def test_it_refuses_a_non_3d_tensor():
    with pytest.raises(ValueError, match="C, H, W"):
        describe_cell(np.zeros((16, 16)), ["a"])


# ---------------------------------------------------------------------------
# Unit-class composition
# ---------------------------------------------------------------------------


def test_unit_class_composition_sums_to_one_over_the_vocabulary():
    cell = synthetic_cell()
    labels = np.zeros((16, 16), dtype=int)
    labels[:8] = 1
    labels[:4] = 2
    d = describe_cell(cell, NAMES, unit_labels=labels, unit_classes=[0, 1, 2])
    assert d["unitclass_0_fraction"] == pytest.approx(0.5)
    assert d["unitclass_1_fraction"] == pytest.approx(0.25)
    assert d["unitclass_2_fraction"] == pytest.approx(0.25)


def test_a_class_absent_from_a_cell_reports_zero_rather_than_vanishing():
    """Otherwise two cells' descriptors would have different lengths."""
    cell = synthetic_cell()
    labels = np.zeros((16, 16), dtype=int)
    d = describe_cell(cell, NAMES, unit_labels=labels, unit_classes=[0, 1, 2])
    assert d["unitclass_1_fraction"] == 0.0
    assert set(d) == set(descriptor_keys(NAMES, unit_classes=[0, 1, 2]))


def test_labels_without_a_vocabulary_are_refused():
    cell = synthetic_cell()
    with pytest.raises(ValueError, match="unit_classes"):
        describe_cell(cell, NAMES, unit_labels=np.zeros((16, 16), dtype=int))


def test_labels_of_the_wrong_shape_are_refused():
    with pytest.raises(ValueError, match="does not match"):
        describe_cell(
            synthetic_cell(), NAMES,
            unit_labels=np.zeros((8, 8), dtype=int), unit_classes=[0],
        )


# ---------------------------------------------------------------------------
# Many cells
# ---------------------------------------------------------------------------


def test_describe_cells_builds_a_frame_indexed_by_cell_id():
    cells = [synthetic_cell(seed=s) for s in range(4)]
    ids = [f"15_{26040 + i}_15320" for i in range(4)]
    frame = describe_cells(cells, NAMES, cell_ids=ids)

    assert list(frame.index) == ids
    assert frame.index.name == "cell_id"
    assert set(frame.columns) == set(descriptor_keys(NAMES))
    assert len(frame) == 4


def test_describe_cells_without_ids_still_works():
    """Generated cells have no id, because they are not anywhere."""
    frame = describe_cells(np.stack([synthetic_cell(seed=s) for s in range(3)]), NAMES)
    assert len(frame) == 3


def test_real_and_generated_frames_share_their_columns():
    real = describe_cells([synthetic_cell(seed=s) for s in range(3)], NAMES)
    rng = np.random.default_rng(0)
    generated = describe_cells(rng.normal(0, 1, (3, 3, 16, 16)), NAMES)
    assert list(real.columns) == list(generated.columns)


def test_a_unit_labeller_is_applied_per_cell():
    cells = [synthetic_cell(seed=s) for s in range(3)]

    def labeller(cell):
        # A stand-in for a fitted C3 classifier: split on the building grid.
        return (cell[0] > 0.5).astype(int)

    frame = describe_cells(cells, NAMES, unit_labeller=labeller, unit_classes=[0, 1])
    assert "unitclass_1_fraction" in frame.columns
    assert np.allclose(
        frame["unitclass_0_fraction"] + frame["unitclass_1_fraction"], 1.0
    )


def test_mismatched_id_count_is_refused():
    with pytest.raises(ValueError, match="cell ids"):
        describe_cells([synthetic_cell()], NAMES, cell_ids=["a", "b"])
