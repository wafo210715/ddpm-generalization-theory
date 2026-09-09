"""Tests for the PCA + UMAP cell map (C4b, E3b).

The property the whole design rests on: fit once on real cells, then place
generated cells on that same map at every checkpoint.  A map refitted per
checkpoint would make the film in E3b meaningless.

Run: pytest tests/test_embedding.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from analysis.embedding import CellEmbedding, flatten_cells


def clustered_cells(n_per_group=20, c=3, size=8, seed=0):
    """Three visibly different groups of cells: a stand-in for typologies."""
    rng = np.random.default_rng(seed)
    groups = []
    for offset in (-4.0, 0.0, 4.0):
        cells = rng.normal(0.0, 0.3, (n_per_group, c, size, size))
        cells[:, 0] += offset
        groups.append(cells)
    labels = np.repeat([0, 1, 2], n_per_group)
    return np.concatenate(groups), labels


# ---------------------------------------------------------------------------
# Flattening
# ---------------------------------------------------------------------------


def test_flatten_gives_one_vector_per_cell():
    cells, _ = clustered_cells()
    assert flatten_cells(cells).shape == (60, 3 * 8 * 8)


def test_flatten_can_take_a_single_grid():
    """C4b: "one map for the whole stack and one per grid"."""
    cells, _ = clustered_cells()
    assert flatten_cells(cells, channel=1).shape == (60, 64)
    assert flatten_cells(cells, channel=[0, 2]).shape == (60, 128)


def test_pooling_shrinks_the_grid_by_the_factor():
    cells, _ = clustered_cells(size=8)
    assert flatten_cells(cells, pool=2).shape == (60, 3 * 4 * 4)


def test_pooling_averages_rather_than_subsamples():
    cell = np.arange(16, dtype=float).reshape(1, 1, 4, 4)
    pooled = flatten_cells(cell, pool=2).reshape(2, 2)
    assert pooled[0, 0] == pytest.approx(np.mean([0, 1, 4, 5]))


def test_pooling_crops_a_remainder_rather_than_failing():
    cells, _ = clustered_cells(size=8)
    assert flatten_cells(cells, pool=3).shape == (60, 3 * 2 * 2)


def test_a_pool_larger_than_the_grid_is_refused():
    cells, _ = clustered_cells(size=8)
    with pytest.raises(ValueError, match="larger than"):
        flatten_cells(cells, pool=16)


def test_flatten_accepts_a_list_of_cells():
    cells, _ = clustered_cells()
    assert np.allclose(flatten_cells(list(cells)), flatten_cells(cells))


# ---------------------------------------------------------------------------
# Fitting and projecting
# ---------------------------------------------------------------------------


def test_the_map_is_two_dimensional():
    cells, _ = clustered_cells()
    embedding = CellEmbedding.fit(cells, n_pca=10)
    assert embedding.transform(cells).shape == (60, 2)


def test_projecting_the_fitting_cells_reproduces_their_positions():
    """The C4b picture has to be stable, or E3b compares against a moving map."""
    cells, _ = clustered_cells()
    embedding = CellEmbedding.fit(cells, n_pca=10)
    first = embedding.transform(cells)
    second = embedding.transform(cells)
    assert np.allclose(first, second)


def test_new_cells_can_be_placed_on_a_map_they_were_not_fitted_on():
    """The reason for UMAP over t-SNE, as a test."""
    cells, _ = clustered_cells()
    embedding = CellEmbedding.fit(cells, n_pca=10)
    generated, _ = clustered_cells(n_per_group=5, seed=99)
    assert embedding.transform(generated).shape == (15, 2)


def test_generated_cells_from_one_group_land_near_that_group():
    """E3b's "generated points that fill a real cluster mean it was learned"."""
    cells, labels = clustered_cells()
    embedding = CellEmbedding.fit(cells, n_pca=10)
    real_xy = embedding.transform(cells)

    rng = np.random.default_rng(5)
    like_group_2 = rng.normal(0.0, 0.3, (10, 3, 8, 8))
    like_group_2[:, 0] += 4.0
    generated_xy = embedding.transform(like_group_2)

    centres = np.stack([real_xy[labels == g].mean(axis=0) for g in (0, 1, 2)])
    nearest = np.argmin(
        np.linalg.norm(generated_xy[:, None, :] - centres[None, :, :], axis=2), axis=1
    )
    assert np.mean(nearest == 2) > 0.8


def test_a_per_grid_map_only_sees_its_own_grid():
    cells, _ = clustered_cells()
    embedding = CellEmbedding.fit(cells, channel=0, n_pca=5)
    assert embedding.channel == 0
    assert embedding.n_features == 64


def test_the_map_refuses_cells_of_the_wrong_shape():
    """Both sides must share a map, so a silent reshape is not acceptable."""
    cells, _ = clustered_cells(c=3, size=8)
    embedding = CellEmbedding.fit(cells, n_pca=5)
    other, _ = clustered_cells(c=6, size=8)
    with pytest.raises(ValueError, match="share a map"):
        embedding.transform(other)


def test_it_reports_how_much_spread_the_pca_step_kept():
    """The cumulative sum of this is C4's k95."""
    cells, _ = clustered_cells()
    embedding = CellEmbedding.fit(cells, n_pca=10)
    ratios = embedding.explained_variance_ratio
    assert ratios.shape == (10,)
    assert 0 < ratios.sum() <= 1.0
    assert np.all(np.diff(ratios) <= 1e-12)


def test_n_pca_is_capped_at_what_the_data_supports():
    """A small pilot set must not raise just because 50 was the default."""
    cells, _ = clustered_cells(n_per_group=2)
    embedding = CellEmbedding.fit(cells, n_pca=50)
    assert embedding.explained_variance_ratio.shape[0] <= 5


def test_too_few_cells_to_fit_a_map_is_refused_clearly():
    cells, _ = clustered_cells()
    with pytest.raises(ValueError, match="at least 3 cells"):
        CellEmbedding.fit(cells[:2])


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_a_saved_map_places_new_points_exactly_where_the_original_did(tmp_path):
    """Checkpoints are analysed over days; the map has to survive on disk."""
    cells, _ = clustered_cells()
    embedding = CellEmbedding.fit(cells, n_pca=10)
    generated, _ = clustered_cells(n_per_group=4, seed=11)
    before = embedding.transform(generated)

    path = str(tmp_path / "maps" / "all_channels.pkl")
    embedding.save(path)
    after = CellEmbedding.load(path).transform(generated)

    assert np.allclose(before, after)


def test_two_fits_with_the_same_seed_agree():
    cells, _ = clustered_cells()
    a = CellEmbedding.fit(cells, n_pca=10, random_state=7).transform(cells)
    b = CellEmbedding.fit(cells, n_pca=10, random_state=7).transform(cells)
    assert np.allclose(a, b)
