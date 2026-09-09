"""Multi-channel cell tensor storage and loader.

Implements B8 of the v4 pipeline map:

    Store cells as multi-channel tensors, not RGB PNGs.  The U-Net reads
    its channel count from the config; one channel or six both work.
    Only the PNG loader forces three.

A store is a directory:

    <root>/
        manifest.json           channel names, size, dtype, fitted pipeline
        cells/<cell_id>.npy     one (C, H, W) array per cell, in data space
        splits/<split>_cells.txt

Cells are held on disk in **data space** --- physical units, before the B11
transform --- so that refitting the transform, or changing which transform a
channel gets, never means re-rasterizing.  The loader applies
:class:`~datasets.channel_transforms.ChannelPipeline` on the way out, and E1
unpacks a generated sample with the same pipeline in reverse.

Usage
-----
    store = CellTensorStore.create(root, channel_names=[...], img_size=1024)
    store.write_cell("15_26043_15320", array)          # (C, H, W)
    store.fit_pipeline(transforms=["log1p", "identity", ...])
    store.write_splits(seed=42)

    ds = CellTensorDataset(root, split="train")        # standardized tensors
    ds = CellTensorDataset(root, split="train", channels=["building_binary"])
"""

from __future__ import annotations

import json
import os
from typing import Iterable, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.channel_transforms import ChannelPipeline

MANIFEST_NAME = "manifest.json"
CELLS_DIRNAME = "cells"
SPLITS_DIRNAME = "splits"
PIPELINE_KEY = "pipeline"

SPLITS = ("train", "val", "test")


class CellTensorStore:
    """A directory of per-cell ``(C, H, W)`` tensors with a manifest.

    One ``.npy`` per cell rather than one big array: cells are written by the
    rasterizer one at a time, the set grows as tiles are added, and every
    dataset mutant in C7 is a different subset of the same files, so nothing
    is copied when a mutant is defined.

    Storage dtype is a real trade-off at this resolution.  A 1024 px
    six-channel cell is 24 MB in float32 and 12 MB in float16, so a
    ten-thousand-cell dataset is 240 GB against 120 GB.  float16 carries about
    three decimal digits and overflows above 65504, which a POI count channel
    could reach.  The default here is float32, because a silently degraded
    indicator is not recoverable and disk is; pass ``dtype="float16"`` to halve
    it once B10's marginals show the dynamic ranges involved.  Either way
    :meth:`write_cell` refuses a cell the dtype cannot represent.
    """

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        manifest_path = os.path.join(self.root, MANIFEST_NAME)
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"no {MANIFEST_NAME} in {self.root}; use CellTensorStore.create()"
            )
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest: dict = json.load(f)

    # -- construction -------------------------------------------------------

    @classmethod
    def create(
        cls,
        root: str,
        channel_names: Sequence[str],
        img_size: int,
        dtype: str = "float32",
        metadata: dict | None = None,
        exist_ok: bool = False,
    ) -> "CellTensorStore":
        """Create an empty store and write its manifest."""
        root = os.path.abspath(root)
        manifest_path = os.path.join(root, MANIFEST_NAME)
        if os.path.exists(manifest_path) and not exist_ok:
            raise FileExistsError(f"{manifest_path} already exists")
        if len(set(channel_names)) != len(channel_names):
            raise ValueError(f"duplicate channel names: {list(channel_names)}")
        if dtype not in ("float16", "float32", "float64"):
            raise ValueError(f"unsupported dtype {dtype!r}")

        os.makedirs(os.path.join(root, CELLS_DIRNAME), exist_ok=True)
        manifest = {
            "channel_names": list(channel_names),
            "img_channels": len(channel_names),
            "img_size": int(img_size),
            "dtype": dtype,
            "space": "data",
            PIPELINE_KEY: None,
            "metadata": metadata or {},
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return cls(root)

    # -- manifest accessors -------------------------------------------------

    @property
    def channel_names(self) -> list[str]:
        return list(self.manifest["channel_names"])

    @property
    def img_channels(self) -> int:
        return int(self.manifest["img_channels"])

    @property
    def img_size(self) -> int:
        return int(self.manifest["img_size"])

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.manifest["dtype"])

    @property
    def cells_dir(self) -> str:
        return os.path.join(self.root, CELLS_DIRNAME)

    @property
    def splits_dir(self) -> str:
        return os.path.join(self.root, SPLITS_DIRNAME)

    def _save_manifest(self) -> None:
        with open(os.path.join(self.root, MANIFEST_NAME), "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2)

    # -- cells --------------------------------------------------------------

    def cell_path(self, cell_id: str) -> str:
        return os.path.join(self.cells_dir, f"{cell_id}.npy")

    def cell_ids(self) -> list[str]:
        """Every cell in the store, sorted, so any split built on it is stable."""
        if not os.path.isdir(self.cells_dir):
            return []
        return sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(self.cells_dir)
            if f.endswith(".npy")
        )

    def __len__(self) -> int:
        return len(self.cell_ids())

    def write_cell(self, cell_id: str, array: np.ndarray) -> str:
        """Write one cell's ``(C, H, W)`` tensor in data space."""
        a = np.asarray(array)
        expected = (self.img_channels, self.img_size, self.img_size)
        if a.shape != expected:
            raise ValueError(
                f"cell {cell_id!r} has shape {a.shape}, store expects {expected}"
            )
        if not np.all(np.isfinite(a)):
            raise ValueError(f"cell {cell_id!r} contains non-finite values")

        # Check the range before casting rather than after: the cast itself
        # warns and silently produces inf.
        max_representable = float(np.finfo(self.dtype).max)
        peak = float(np.abs(a).max())
        if peak > max_representable:
            raise ValueError(
                f"cell {cell_id!r} overflows {self.dtype} (max |value| "
                f"{peak:.4g} > {max_representable:.4g}); create the store "
                f"with dtype='float32'"
            )
        stored = a.astype(self.dtype)
        os.makedirs(self.cells_dir, exist_ok=True)
        path = self.cell_path(cell_id)
        np.save(path, stored)
        return path

    def read_cell(self, cell_id: str, mmap: bool = False) -> np.ndarray:
        """Read one cell's tensor, in data space, as float32."""
        arr = np.load(self.cell_path(cell_id), mmap_mode="r" if mmap else None)
        return np.asarray(arr, dtype=np.float32)

    def iter_cells(self, cell_ids: Iterable[str] | None = None) -> Iterator[np.ndarray]:
        for cell_id in cell_ids if cell_ids is not None else self.cell_ids():
            yield self.read_cell(cell_id)

    # -- pipeline -----------------------------------------------------------

    @property
    def pipeline(self) -> ChannelPipeline | None:
        """The fitted B11 transform + standardization, or None if not fitted."""
        stored = self.manifest.get(PIPELINE_KEY)
        return ChannelPipeline.from_dict(stored) if stored else None

    def set_pipeline(self, pipeline: ChannelPipeline) -> None:
        """Store a pipeline in the manifest.  It is the decoder for E1."""
        if pipeline.channel_names != self.channel_names:
            raise ValueError(
                f"pipeline channels {pipeline.channel_names} do not match store "
                f"channels {self.channel_names}"
            )
        self.manifest[PIPELINE_KEY] = pipeline.to_dict()
        self._save_manifest()

    def fit_pipeline(
        self,
        transforms: Sequence[str] | str = "identity",
        cell_ids: Sequence[str] | None = None,
        max_cells: int | None = None,
        seed: int = 42,
    ) -> ChannelPipeline:
        """Fit and store the pipeline.

        Fit on the training split only when one exists, so nothing about the
        held-out cells leaks into the standardization.  ``max_cells``
        subsamples, since the marginals converge long before every cell is read.
        """
        if cell_ids is None:
            cell_ids = self._fit_cell_ids()
        cell_ids = list(cell_ids)
        if not cell_ids:
            raise ValueError("no cells to fit the pipeline on")
        if max_cells is not None and len(cell_ids) > max_cells:
            rng = np.random.default_rng(seed)
            cell_ids = [cell_ids[i] for i in
                        sorted(rng.choice(len(cell_ids), max_cells, replace=False))]

        pipeline = ChannelPipeline.fit(
            self.iter_cells(cell_ids),
            channel_names=self.channel_names,
            transforms=transforms,
        )
        self.set_pipeline(pipeline)
        return pipeline

    def _fit_cell_ids(self) -> list[str]:
        train_split = os.path.join(self.splits_dir, "train_cells.txt")
        if os.path.exists(train_split):
            return self.read_split("train")
        return self.cell_ids()

    # -- splits -------------------------------------------------------------

    def write_splits(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        cell_ids: Sequence[str] | None = None,
    ) -> dict[str, list[str]]:
        """Write a deterministic train/val/test split, seeded and reproducible."""
        if not 0 < train_ratio + val_ratio < 1.0:
            raise ValueError("train_ratio + val_ratio must be in (0, 1)")
        ids = list(cell_ids) if cell_ids is not None else self.cell_ids()
        if not ids:
            raise ValueError("no cells to split")

        rng = np.random.default_rng(seed)
        order = rng.permutation(len(ids))
        n_train = int(len(ids) * train_ratio)
        n_val = int(len(ids) * val_ratio)
        splits = {
            "train": [ids[i] for i in order[:n_train]],
            "val": [ids[i] for i in order[n_train : n_train + n_val]],
            "test": [ids[i] for i in order[n_train + n_val :]],
        }

        os.makedirs(self.splits_dir, exist_ok=True)
        for split, members in splits.items():
            with open(
                os.path.join(self.splits_dir, f"{split}_cells.txt"), "w", encoding="utf-8"
            ) as f:
                f.write("\n".join(members) + ("\n" if members else ""))
        return splits

    def read_split(self, split: str) -> list[str]:
        if split == "all":
            return self.cell_ids()
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}; expected one of {SPLITS} or 'all'")
        path = os.path.join(self.splits_dir, f"{split}_cells.txt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"no split file {path}; call write_splits() first")
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]


class CellTensorDataset(Dataset):
    """Loads cell tensors for any channel count.

    Returns ``{"tensor": (C, H, W) float32, "cell_id": str}``.  The ``tensor``
    key rather than ``image``: the cell is a stack of indicator grids, and the
    only thing in it that was ever an image was the pilot's container format.

    Parameters
    ----------
    root:
        Store directory.
    split:
        ``"train"``, ``"val"``, ``"test"``, or ``"all"``.
    channels:
        Channel names to load, in the order given.  ``None`` loads all of them
        in store order.  This is how a per-grid analysis (E3b) or a
        single-channel ablation gets its data without a second store.
    standardize:
        Apply the stored pipeline.  Training wants this on; a coverage report
        reading physical units wants it off.
    cell_ids:
        An explicit list, overriding ``split``.  This is how a C7 mutant is
        trained: a subset of the same store, named rather than copied.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        channels: Sequence[str] | None = None,
        standardize: bool = True,
        cell_ids: Sequence[str] | None = None,
        mmap: bool = False,
    ):
        super().__init__()
        self.store = CellTensorStore(root)
        self.split = split
        self.standardize = standardize
        self.mmap = mmap

        if cell_ids is not None:
            self.cell_ids = list(cell_ids)
            missing = [c for c in self.cell_ids if not os.path.exists(self.store.cell_path(c))]
            if missing:
                raise FileNotFoundError(f"{len(missing)} cell ids not in store, e.g. {missing[:3]}")
        else:
            self.cell_ids = self.store.read_split(split)
        if not self.cell_ids:
            raise ValueError(f"split {split!r} of {root} is empty")

        store_names = self.store.channel_names
        self.channel_names = list(channels) if channels is not None else store_names
        unknown = [c for c in self.channel_names if c not in store_names]
        if unknown:
            raise KeyError(f"no such channels in store: {unknown}; have {store_names}")
        self.channel_indices = np.array(
            [store_names.index(c) for c in self.channel_names], dtype=np.intp
        )

        self.pipeline: ChannelPipeline | None = None
        if standardize:
            full = self.store.pipeline
            if full is None:
                raise ValueError(
                    f"{root} has no fitted pipeline; call store.fit_pipeline() "
                    f"or pass standardize=False"
                )
            self.pipeline = full.subset(self.channel_names)

    @property
    def img_channels(self) -> int:
        """The channel count to hand the U-Net."""
        return len(self.channel_names)

    @property
    def img_size(self) -> int:
        return self.store.img_size

    def __len__(self) -> int:
        return len(self.cell_ids)

    def __getitem__(self, idx: int) -> dict:
        cell_id = self.cell_ids[idx]
        arr = self.store.read_cell(cell_id, mmap=self.mmap)[self.channel_indices]
        if self.pipeline is not None:
            arr = self.pipeline.forward(arr, dtype=np.float32)
        return {"tensor": torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)),
                "cell_id": cell_id}

    def unpack(self, tensor: torch.Tensor | np.ndarray) -> np.ndarray:
        """E1: a standardized ``(C, H, W)`` or ``(N, C, H, W)`` back to data space.

        The same pipeline in reverse, so a generated sample and a real cell are
        read in the same physical units.
        """
        arr = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
        if self.pipeline is None:
            return arr.astype(np.float64)
        return self.pipeline.inverse(arr)
