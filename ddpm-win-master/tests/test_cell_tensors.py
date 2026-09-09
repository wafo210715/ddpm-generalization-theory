"""Tests for multi-channel cell tensor storage and loading (B8).

Synthetic cells only: the point is that the data path does not care how many
channels there are, so the tests sweep the channel count rather than assume
three.

Run: pytest tests/test_cell_tensors.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch

from datasets.cell_tensors import CellTensorDataset, CellTensorStore
from datasets.dataset import get_dataset


def make_store(root, n_cells=20, n_channels=6, img_size=8, dtype="float32", seed=0):
    """A store of synthetic cells with names in the shape of the real ones."""
    rng = np.random.default_rng(seed)
    names = [f"channel_{i}" for i in range(n_channels)]
    store = CellTensorStore.create(root, names, img_size=img_size, dtype=dtype)
    for i in range(n_cells):
        # Cell ids in the zoom-15 tile form: z_x_y
        cell_id = f"15_{26040 + i}_15320"
        store.write_cell(cell_id, rng.random((n_channels, img_size, img_size)))
    return store


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_channels", [1, 3, 6, 14])
def test_store_round_trips_any_channel_count(tmp_path, n_channels):
    """B8: "one channel or six both work"; 14 is the companion page's upper case."""
    store = make_store(tmp_path / "store", n_channels=n_channels)
    assert store.img_channels == n_channels
    assert len(store) == 20
    cell = store.read_cell(store.cell_ids()[0])
    assert cell.shape == (n_channels, 8, 8)


def test_store_rejects_a_wrong_shaped_cell(tmp_path):
    store = make_store(tmp_path / "store", n_channels=6)
    with pytest.raises(ValueError, match="shape"):
        store.write_cell("bad", np.zeros((3, 8, 8)))


def test_store_rejects_non_finite_values(tmp_path):
    store = make_store(tmp_path / "store")
    bad = np.zeros((6, 8, 8))
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        store.write_cell("bad", bad)


def test_float16_store_refuses_a_cell_it_would_silently_destroy(tmp_path):
    """float16 halves the 24 MB a 1024 px six-channel cell costs; it also
    overflows above 65504, and a POI count channel could reach that."""
    store = CellTensorStore.create(tmp_path / "s", ["a"], img_size=4, dtype="float16")
    with pytest.raises(ValueError, match="float32"):
        store.write_cell("big", np.full((1, 4, 4), 1e6))


def test_manifest_survives_reopening(tmp_path):
    root = tmp_path / "store"
    make_store(root, n_channels=5)
    reopened = CellTensorStore(str(root))
    assert reopened.channel_names == [f"channel_{i}" for i in range(5)]
    assert reopened.img_size == 8


def test_creating_over_an_existing_store_is_refused(tmp_path):
    root = tmp_path / "store"
    make_store(root)
    with pytest.raises(FileExistsError):
        CellTensorStore.create(root, ["a"], img_size=8)


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def test_splits_are_deterministic_disjoint_and_complete(tmp_path):
    store = make_store(tmp_path / "store", n_cells=100)
    first = store.write_splits(seed=42)
    second = store.write_splits(seed=42)
    assert first == second

    members = first["train"] + first["val"] + first["test"]
    assert len(members) == len(set(members)) == 100
    assert set(members) == set(store.cell_ids())
    assert len(first["train"]) == 80 and len(first["val"]) == 10


def test_a_different_seed_gives_a_different_split(tmp_path):
    store = make_store(tmp_path / "store", n_cells=100)
    assert store.write_splits(seed=1)["train"] != store.write_splits(seed=2)["train"]


# ---------------------------------------------------------------------------
# Pipeline attached to the store
# ---------------------------------------------------------------------------


def test_pipeline_is_stored_in_the_manifest_and_reloads(tmp_path):
    root = tmp_path / "store"
    store = make_store(root)
    store.write_splits()
    fitted = store.fit_pipeline(transforms="log1p")

    reopened = CellTensorStore(str(root))
    assert reopened.pipeline is not None
    assert reopened.pipeline.channel_names == fitted.channel_names
    cell = reopened.read_cell(reopened.cell_ids()[0]).astype(np.float64)
    assert np.abs(reopened.pipeline.inverse(reopened.pipeline.forward(cell)) - cell).max() < 1e-9


def test_pipeline_is_fitted_on_the_training_split_only(tmp_path):
    """Nothing about held-out cells may leak into the standardization."""
    root = tmp_path / "store"
    store = make_store(root, n_cells=40)
    store.write_splits(seed=7)
    from_store = store.fit_pipeline(transforms="identity")

    from datasets.channel_transforms import ChannelPipeline

    explicit = ChannelPipeline.fit(
        store.iter_cells(store.read_split("train")), store.channel_names, "identity"
    )
    assert from_store.channels[0].mean == pytest.approx(explicit.channels[0].mean)

    all_cells = ChannelPipeline.fit(
        store.iter_cells(), store.channel_names, "identity"
    )
    assert from_store.channels[0].mean != pytest.approx(all_cells.channels[0].mean)


def test_pipeline_rejects_channels_that_do_not_match_the_store(tmp_path):
    from datasets.channel_transforms import ChannelPipeline

    store = make_store(tmp_path / "store", n_channels=3)
    wrong = ChannelPipeline.fit(np.random.rand(4, 3, 8, 8), ["x", "y", "z"], "identity")
    with pytest.raises(ValueError, match="do not match"):
        store.set_pipeline(wrong)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_channels", [1, 3, 6, 14])
def test_dataset_yields_tensors_of_the_store_s_channel_count(tmp_path, n_channels):
    root = tmp_path / "store"
    store = make_store(root, n_channels=n_channels)
    store.write_splits()
    store.fit_pipeline(transforms="identity")

    ds = CellTensorDataset(str(root), split="train")
    item = ds[0]
    assert isinstance(item["tensor"], torch.Tensor)
    assert item["tensor"].shape == (n_channels, 8, 8)
    assert item["tensor"].dtype == torch.float32
    assert ds.img_channels == n_channels
    assert item["cell_id"] in store.read_split("train")


def test_dataset_channel_subset_selects_and_reorders(tmp_path):
    root = tmp_path / "store"
    store = make_store(root, n_channels=6)
    store.write_splits()
    store.fit_pipeline(transforms="identity")

    full = CellTensorDataset(str(root), split="train")
    subset = CellTensorDataset(str(root), split="train", channels=["channel_4", "channel_1"])

    assert subset.img_channels == 2
    assert subset.channel_names == ["channel_4", "channel_1"]
    idx = full.cell_ids.index(subset.cell_ids[0])
    assert torch.allclose(subset[0]["tensor"][0], full[idx]["tensor"][4], atol=1e-6)
    assert torch.allclose(subset[0]["tensor"][1], full[idx]["tensor"][1], atol=1e-6)


def test_dataset_standardizes_and_unpacks_back_to_data_space(tmp_path):
    """E1: the sample comes back in physical units through the same pipeline."""
    root = tmp_path / "store"
    store = make_store(root)
    store.write_splits()
    store.fit_pipeline(transforms="log1p")

    ds = CellTensorDataset(str(root), split="train")
    item = ds[0]
    raw = store.read_cell(item["cell_id"]).astype(np.float64)
    assert np.abs(ds.unpack(item["tensor"]) - raw).max() < 1e-5


def test_dataset_without_a_pipeline_returns_data_space(tmp_path):
    root = tmp_path / "store"
    store = make_store(root)
    store.write_splits()

    ds = CellTensorDataset(str(root), split="train", standardize=False)
    item = ds[0]
    raw = store.read_cell(item["cell_id"])
    assert np.abs(item["tensor"].numpy() - raw).max() < 1e-6


def test_standardize_without_a_fitted_pipeline_is_an_error_not_a_silent_pass(tmp_path):
    root = tmp_path / "store"
    store = make_store(root)
    store.write_splits()
    with pytest.raises(ValueError, match="no fitted pipeline"):
        CellTensorDataset(str(root), split="train")


def test_explicit_cell_ids_define_a_mutant_without_copying_anything(tmp_path):
    """C7 mutants are subsets of one store, named rather than duplicated."""
    root = tmp_path / "store"
    store = make_store(root, n_cells=30)
    store.write_splits()
    store.fit_pipeline(transforms="identity")

    mutant = store.cell_ids()[:5]
    ds = CellTensorDataset(str(root), cell_ids=mutant)
    assert len(ds) == 5
    assert [ds[i]["cell_id"] for i in range(5)] == mutant


def test_unknown_channel_name_is_reported_with_the_ones_available(tmp_path):
    root = tmp_path / "store"
    store = make_store(root, n_channels=3)
    store.write_splits()
    store.fit_pipeline()
    with pytest.raises(KeyError, match="no such channels"):
        CellTensorDataset(str(root), split="train", channels=["accessibility"])


def test_dataset_works_through_a_dataloader(tmp_path):
    from torch.utils.data import DataLoader

    root = tmp_path / "store"
    store = make_store(root, n_cells=16, n_channels=6)
    store.write_splits()
    store.fit_pipeline(transforms="identity")

    loader = DataLoader(CellTensorDataset(str(root), split="train"), batch_size=4)
    batch = next(iter(loader))
    assert batch["tensor"].shape == (4, 6, 8, 8)
    assert len(batch["cell_id"]) == 4


def test_get_dataset_factory_reaches_the_store(tmp_path):
    root = tmp_path / "store"
    store = make_store(root, n_channels=6)
    store.write_splits()
    store.fit_pipeline()

    ds = get_dataset("cell_tensors", data_dir=str(root), split="val")
    assert isinstance(ds, CellTensorDataset)
    assert ds.img_channels == 6


def test_get_dataset_says_what_is_missing_without_a_data_dir():
    with pytest.raises(ValueError, match="data_dir"):
        get_dataset("cell_tensors")
