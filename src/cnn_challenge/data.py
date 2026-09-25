"""Dataset construction, train/validation splitting and DataLoaders.

Two details here are deliberate design decisions rather than boilerplate:

1. **Stratified splitting.** The starter notebook uses `random_split`, which with
   only ~150 images per class can easily leave one class with 20 validation
   images and another with 40. That makes a 1% accuracy difference between
   experiments impossible to interpret. A stratified split holds the per-class
   validation count fixed.

2. **Separate transform pipelines per split.** `ImageFolder` attaches one
   transform to the whole dataset, so splitting it with `Subset` would apply
   training augmentation to the validation images too. We instead build two
   `ImageFolder` views over the same directory and index each with the matching
   half of the split.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets

from .transforms import build_transforms
from .utils import seed_worker

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def natural_key(name: str) -> list[object]:
    """Sort key that orders `image_2` before `image_10`.

    Plain lexicographic sorting interleaves them (`image_10` before `image_2`),
    which would silently misalign a predictions file against any externally
    ordered ground truth.
    """
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r"(\d+)", name)]


class FlatImageDataset(Dataset):
    """A directory of unlabelled images, as shipped in this dataset's `test2`.

    `ImageFolder` cannot read it because there are no class subdirectories. Returns
    `(tensor, filename)` so predictions can be written against the source names.
    """

    def __init__(self, root: str | Path, transform=None) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Not a directory: {self.root}")
        self.paths = sorted(
            (p for p in self.root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS),
            key=lambda p: natural_key(p.name),
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found in {self.root}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Any, str]:
        path = self.paths[index]
        # Convert to RGB so grayscale source files still yield 3 channels; the
        # transform pipeline handles any conversion back to 1 channel.
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, path.name


def stratified_split(
    targets: list[int], val_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Split indices so each class contributes the same fraction to validation."""
    rng = np.random.default_rng(seed)
    targets_arr = np.asarray(targets)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for class_id in np.unique(targets_arr):
        class_indices = np.flatnonzero(targets_arr == class_id)
        rng.shuffle(class_indices)
        n_val = int(round(len(class_indices) * val_fraction))
        # Never let a class vanish from either split.
        n_val = max(1, min(n_val, len(class_indices) - 1))
        val_idx.extend(class_indices[:n_val].tolist())
        train_idx.extend(class_indices[n_val:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def random_indices_split(
    n_items: int, val_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Unstratified split, kept so the starter baseline can be reproduced exactly."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_items)
    n_val = int(round(n_items * val_fraction))
    return order[n_val:].tolist(), order[:n_val].tolist()


def build_datasets(cfg: dict[str, Any]) -> dict[str, Any]:
    """Create train/val/test datasets with split-appropriate transforms."""
    data_cfg = cfg["data"]
    root = Path(data_cfg["root"])
    train_root = root / data_cfg["train_dir"]
    test_root = root / data_cfg["test_dir"]

    if not train_root.is_dir():
        raise FileNotFoundError(
            f"Training directory not found: {train_root.resolve()}\n"
            "Run `python scripts/prepare_data.py --zip <downloaded.zip>` first."
        )

    train_tf = build_transforms(data_cfg, cfg["augment"], train=True)
    eval_tf = build_transforms(data_cfg, cfg["augment"], train=False)

    # Same directory, two transform pipelines (see module docstring).
    train_view = datasets.ImageFolder(train_root, transform=train_tf)
    eval_view = datasets.ImageFolder(train_root, transform=eval_tf)

    if data_cfg.get("split", "stratified") == "stratified":
        train_idx, val_idx = stratified_split(
            train_view.targets, float(data_cfg["val_fraction"]), int(cfg["experiment"]["seed"])
        )
    else:
        train_idx, val_idx = random_indices_split(
            len(train_view), float(data_cfg["val_fraction"]), int(cfg["experiment"]["seed"])
        )

    assert not set(train_idx) & set(val_idx), "Train/val split leaked indices"

    train_dataset = Subset(train_view, train_idx)
    val_dataset = Subset(eval_view, val_idx)

    test_dataset = None
    if test_root.is_dir():
        test_dataset = datasets.ImageFolder(test_root, transform=eval_tf)
        if test_dataset.classes != train_view.classes:
            raise ValueError(
                "Test classes do not match train classes; label indices would be wrong.\n"
                f"train: {train_view.classes}\ntest:  {test_dataset.classes}"
            )

    val_targets = [train_view.targets[i] for i in val_idx]
    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
        "class_names": train_view.classes,
        "train_targets": [train_view.targets[i] for i in train_idx],
        "val_targets": val_targets,
        "val_class_counts": dict(sorted(Counter(val_targets).items())),
    }


def build_loaders(cfg: dict[str, Any], bundle: dict[str, Any]) -> dict[str, DataLoader | None]:
    """Wrap datasets in DataLoaders with reproducible shuffling."""
    data_cfg = cfg["data"]
    batch_size = int(data_cfg["batch_size"])
    num_workers = int(data_cfg["num_workers"])
    pin_memory = torch.cuda.is_available()
    generator = torch.Generator()
    generator.manual_seed(int(cfg["experiment"]["seed"]))

    common = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }

    loaders: dict[str, DataLoader | None] = {
        "train": DataLoader(
            bundle["train"],
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=generator,
            **common,
        ),
        "val": DataLoader(bundle["val"], batch_size=batch_size, shuffle=False, **common),
        "test": None,
    }
    if bundle["test"] is not None:
        loaders["test"] = DataLoader(
            bundle["test"], batch_size=batch_size, shuffle=False, **common
        )
    return loaders


def describe_dataset(bundle: dict[str, Any]) -> str:
    """Human-readable split summary, printed at the start of every run."""
    lines = [
        f"Classes ({len(bundle['class_names'])}): {bundle['class_names']}",
        f"Train images: {len(bundle['train'])}",
        f"Val images:   {len(bundle['val'])}",
    ]
    if bundle["test"] is not None:
        lines.append(f"Test images:  {len(bundle['test'])}")
    counts = Counter(bundle["train_targets"])
    per_class = [counts.get(i, 0) for i in range(len(bundle["class_names"]))]
    lines.append(f"Train images per class: min={min(per_class)} max={max(per_class)}")
    return "\n".join(lines)
