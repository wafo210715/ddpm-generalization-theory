"""
Dataset loaders for DDPM training.

Supports:
  - "cell_tensors": multi-channel cell tensors of any channel count (B8 of the
    v4 pipeline map) — see datasets/cell_tensors.py
  - "mnist": MNIST handwritten digits (via torchvision)
  - "geospatial": HCMC urban vitality cell images (PNG from data-processing pipeline)
  - "hcmc_vitality_v2": HCMC urban vitality v2 cell images with non-empty filtering,
    train/val/test splits, and cell_id metadata.

Usage in train.py:
    dataset = get_dataset(config.dataset_name, img_size=config.img_size)
"""

import os
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from PIL import Image
from torchvision import datasets as tv_datasets
from torchvision import transforms

from datasets.cell_tensors import CellTensorDataset


# ---------------------------------------------------------------------------
# Default data directories (relative to ddpm-win-master root)
# ---------------------------------------------------------------------------
_GEOSPATIAL_DEFAULT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    os.pardir,
    "data-processing",
    "data",
    "hcmc",
    "processed",
    "cells_256x256",
)

_HCMC_V2_DEFAULT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    os.pardir,
    "data-processing",
    "data",
    "hcmc",
    "processed",
    "cells",
)


# ===================================================================
#  MNIST Dataset
# ===================================================================


class MNISTDataset(Dataset):
    """Thin wrapper around torchvision MNIST that returns (tensor,) tuples
    and applies standard DDPM preprocessing (ToTensor, resize, center+normalize)."""

    def __init__(
        self,
        img_size: int = 32,
        root: str = "data",
        center: bool = True,
        normalize: bool = False,
        download: bool = True,
    ):
        transform_list = [
            transforms.ToTensor(),
            transforms.Resize(img_size),
        ]
        # Center: [0,1] -> [-0.5, 0.5]
        # Normalize: [-0.5, 0.5] -> [-1, 1]
        if center and normalize:
            # Combined: (x - 0.5) / 0.5 = 2x - 1
            transform_list.append(transforms.Normalize([0.5], [0.5]))
        elif center:
            transform_list.append(transforms.Normalize([0.5], [1.0]))
        elif normalize:
            transform_list.append(transforms.Normalize([0.0], [0.5]))

        self.transform = transforms.Compose(transform_list)
        self.mnist = tv_datasets.MNIST(
            root=root, train=True, download=download, transform=self.transform
        )

    def __len__(self) -> int:
        return len(self.mnist)

    def __getitem__(self, idx: int) -> Tensor:
        img, _label = self.mnist[idx]
        return img


# ===================================================================
#  GeoSpatial Dataset (HCMC urban vitality cells)
# ===================================================================


class GeoSpatialDataset(Dataset):
    """Loads 256x256 RGB PNG cell images from the HCMC data-processing pipeline.

    Channel semantics (for reference):
      B = buildings (binary mask)
      G = POI entropy
      R = accessibility

    Preprocessing:
      1. PIL Image -> float32 tensor in [0, 1] (via ToTensor)
      2. Resize to img_size (identity if already correct)
      3. Center:  subtract 0.5  -> [-0.5, 0.5]
      4. Normalize: divide by 0.5 -> [-1, 1]

    Train / val / test split is deterministic (seeded) and based on sorted
    filenames so it is fully reproducible.
    """

    _SPLIT_SEED = 42

    def __init__(
        self,
        root_dir: str = _GEOSPATIAL_DEFAULT_DIR,
        img_size: int = 256,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        center: bool = True,
        normalize: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        assert split in ("train", "val", "test"), f"Invalid split: {split!r}"
        assert 0 < train_ratio + val_ratio < 1.0, (
            "train_ratio + val_ratio must be in (0, 1)"
        )
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split

        # --- discover all PNG files, sorted deterministically ---
        all_files = sorted(
            f for f in os.listdir(root_dir) if f.lower().endswith(".png")
        )
        if not all_files:
            raise FileNotFoundError(f"No PNG files found in {root_dir}")

        # --- deterministic split ---
        n = len(all_files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # test gets the remainder

        if split == "train":
            self.file_paths = [os.path.join(root_dir, f) for f in all_files[:n_train]]
        elif split == "val":
            self.file_paths = [
                os.path.join(root_dir, f) for f in all_files[n_train : n_train + n_val]
            ]
        else:
            self.file_paths = [
                os.path.join(root_dir, f) for f in all_files[n_train + n_val :]
            ]

        # --- transforms ---
        transform_list = [
            transforms.ToTensor(),  # HWC uint8 -> CHW float32 [0,1]
            transforms.Resize(img_size),  # safety resize (no-op if correct)
        ]
        if center and normalize:
            transform_list.append(
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            )
        elif center:
            transform_list.append(
                transforms.Normalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
            )
        elif normalize:
            transform_list.append(
                transforms.Normalize([0.0, 0.0, 0.0], [0.5, 0.5, 0.5])
            )

        self.transform = transforms.Compose(transform_list)

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tensor:
        img_path = self.file_paths[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img)  # type: ignore[return-value]


# ===================================================================
#  HCMC Vitality v2 Dataset (non-empty filtering + splits + cell_id)
# ===================================================================


class HCMCVitalityV2Dataset(Dataset):
    """HCMC urban vitality v2 — 128x128 RGB PNG cell images.

    Key differences from GeoSpatialDataset:
      - Filters to non-empty cells (max pixel value > 10)
      - Saves reproducible train/val/test split files
      - Returns dict {'image': Tensor, 'cell_id': str}

    Channel order: R=accessibility, G=diversity, B=density

    Preprocessing:
      1. PIL Image -> float32 tensor in [0, 1] (via ToTensor)
      2. Resize to img_size (bicubic, identity if already 128)
      3. Normalize to [-1, 1]

    Split: 80/10/10 (deterministic, seeded). Split files saved to
    ``splits/{split}_cells.txt`` under the data root.
    """

    _SPLIT_SEED = 42
    _EMPTY_THRESHOLD = 10  # max pixel value <= 10 means empty

    def __init__(
        self,
        root_dir: str = _HCMC_V2_DEFAULT_DIR,
        img_size: int = 128,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        center: bool = True,
        normalize: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        assert split in ("train", "val", "test"), f"Invalid split: {split!r}"
        assert 0 < train_ratio + val_ratio < 1.0, (
            "train_ratio + val_ratio must be in (0, 1)"
        )
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split

        # --- discover all PNG files, sorted deterministically ---
        all_files = sorted(
            f for f in os.listdir(root_dir) if f.lower().endswith(".png")
        )
        if not all_files:
            raise FileNotFoundError(f"No PNG files found in {root_dir}")

        # --- filter non-empty cells ---
        non_empty = []
        for fname in all_files:
            fpath = os.path.join(root_dir, fname)
            img = Image.open(fpath).convert("RGB")
            arr = np.array(img)
            if arr.max() > self._EMPTY_THRESHOLD:
                non_empty.append(fname)
            img.close()

        # --- load or create deterministic split ---
        split_dir = os.path.join(root_dir, "splits")
        split_file = os.path.join(split_dir, f"{split}_cells.txt")

        if os.path.exists(split_file):
            # Load existing split
            with open(split_file, "r", encoding="utf-8") as f:
                cell_ids = [line.strip() for line in f if line.strip()]
            self.file_names = cell_ids
        else:
            # Create and save splits
            rng = np.random.default_rng(seed)
            indices = np.arange(len(non_empty))
            rng.shuffle(indices)

            n = len(non_empty)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            splits = {
                "train": [non_empty[i] for i in indices[:n_train]],
                "val": [non_empty[i] for i in indices[n_train : n_train + n_val]],
                "test": [non_empty[i] for i in indices[n_train + n_val :]],
            }

            os.makedirs(split_dir, exist_ok=True)
            for split_name, file_list in splits.items():
                path = os.path.join(split_dir, f"{split_name}_cells.txt")
                with open(path, "w", encoding="utf-8") as f:
                    for fname in file_list:
                        # Write cell_id (filename without .png)
                        f.write(fname + "\n")

            self.file_names = splits[split]

        self.file_paths = [os.path.join(root_dir, f) for f in self.file_names]

        # --- transforms ---
        transform_list = [
            transforms.ToTensor(),  # HWC uint8 -> CHW float32 [0,1]
        ]
        if img_size != 128:
            transform_list.append(
                transforms.Resize(
                    img_size, interpolation=transforms.InterpolationMode.BICUBIC
                )
            )

        if center and normalize:
            transform_list.append(
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            )
        elif center:
            transform_list.append(
                transforms.Normalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
            )
        elif normalize:
            transform_list.append(
                transforms.Normalize([0.0, 0.0, 0.0], [0.5, 0.5, 0.5])
            )

        self.transform = transforms.Compose(transform_list)

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        img_path = self.file_paths[idx]
        cell_id = os.path.splitext(self.file_names[idx])[0]
        img = Image.open(img_path).convert("RGB")
        tensor = self.transform(img)
        return {"image": tensor, "cell_id": cell_id}


# ===================================================================
#  Factory function — the interface expected by train.py
# ===================================================================


def get_dataset(
    name: str,
    img_size: int = 32,
    data_dir: Optional[str] = None,
    split: str = "train",
    center: bool = True,
    normalize: bool = True,
    **kwargs,
) -> Dataset:
    """Return a PyTorch Dataset suitable for DDPM training.

    Parameters
    ----------
    name : str
        Dataset identifier: "cell_tensors", "mnist", "geospatial", or
        "hcmc_vitality_v2".
    img_size : int
        Target image size (images will be resized to img_size x img_size).
    data_dir : str, optional
        Root directory for the dataset.  Falls back to a sensible default
        when *None*.
    split : str
        One of "train", "val", "test".  Only used by GeoSpatialDataset
        and HCMCVitalityV2Dataset.
    center : bool
        Subtract 0.5 so the tensor is centered around zero.
    normalize : bool
        Scale to [-1, 1] (implies centering).

    Returns
    -------
    Dataset  that yields (C, H, W) float32 tensors or dicts.
    """
    name = name.lower()

    if name == "cell_tensors":
        # Multi-channel store (B8).  Channel count comes from the store's
        # manifest, not from a hard-coded 1 or 3, and `channels` selects a
        # subset.  Standardization is the fitted B11 pipeline, so the
        # center/normalize flags of the PNG loaders do not apply.
        if data_dir is None:
            raise ValueError(
                "cell_tensors needs data_dir (or data_root in the config) "
                "pointing at the store directory"
            )
        return CellTensorDataset(
            root=data_dir,
            split=split,
            channels=kwargs.get("channels"),
            standardize=kwargs.get("standardize", True),
            cell_ids=kwargs.get("cell_ids"),
            mmap=kwargs.get("mmap", False),
        )

    if name == "mnist":
        return MNISTDataset(
            img_size=img_size,
            root=data_dir or "data",
            center=center,
            normalize=normalize,
        )

    if name == "geospatial":
        return GeoSpatialDataset(
            root_dir=data_dir or _GEOSPATIAL_DEFAULT_DIR,
            img_size=img_size,
            split=split,
            center=center,
            normalize=normalize,
        )

    if name == "hcmc_vitality_v2":
        return HCMCVitalityV2Dataset(
            root_dir=data_dir or _HCMC_V2_DEFAULT_DIR,
            img_size=img_size,
            split=split,
            center=center,
            normalize=normalize,
        )

    raise ValueError(
        f"Unknown dataset: {name!r}.  Supported: 'cell_tensors', 'mnist', "
        f"'geospatial', 'hcmc_vitality_v2'."
    )
