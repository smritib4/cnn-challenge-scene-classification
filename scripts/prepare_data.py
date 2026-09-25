"""Unpack the downloaded dataset into the layout the training code expects.

    python scripts/prepare_data.py --zip "C:/Users/you/Downloads/dataset.zip"
    python scripts/prepare_data.py --src "C:/Users/you/Downloads/extracted_folder"

Target layout:

    data/
      train/<class_name>/*.jpg
      test/<class_name>/*.jpg

The script locates the `train`/`test` directories wherever they sit inside the
archive, ignores macOS metadata entries, verifies that both splits share the same
class names, and prints per-class counts so a truncated download is obvious
before any GPU time is spent.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_ALIASES = {
    "train": {"train", "training", "train_set"},
    "test": {"test", "testing", "test_set", "val", "validation"},
}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTENSIONS and not path.name.startswith(".")


def extract_zip(zip_path: Path, dest: Path) -> Path:
    """Extract a zip, skipping macOS resource-fork entries."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        members = [
            m
            for m in archive.namelist()
            if not m.startswith("__MACOSX") and "/._" not in m and not m.split("/")[-1].startswith("._")
        ]
        archive.extractall(dest, members=members)
    print(f"Extracted {len(members)} entries to {dest}")
    return dest


def find_split_dirs(root: Path) -> dict[str, Path]:
    """Search the tree for directories that look like the train/test splits."""
    found: dict[str, Path] = {}
    for candidate in [root, *sorted(p for p in root.rglob("*") if p.is_dir())]:
        name = candidate.name.lower()
        for split, aliases in SPLIT_ALIASES.items():
            if name in aliases and split not in found:
                # A split directory must contain class subdirectories with images.
                subdirs = [d for d in candidate.iterdir() if d.is_dir()]
                if subdirs and any(any(is_image(f) for f in d.rglob("*")) for d in subdirs):
                    found[split] = candidate
    return found


def copy_split(src: Path, dest: Path) -> Counter:
    """Copy class folders into `dest`, returning per-class image counts."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    for class_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        target_dir = dest / class_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)
        for image in sorted(class_dir.rglob("*")):
            if is_image(image):
                shutil.copy2(image, target_dir / image.name)
                counts[class_dir.name] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--zip", type=str, help="Path to the downloaded dataset .zip")
    group.add_argument("--src", type=str, help="Path to an already-extracted dataset folder")
    parser.add_argument("--out", type=str, default="data", help="Destination root (default: data)")
    parser.add_argument(
        "--keep-staging", action="store_true", help="Keep the intermediate extraction folder"
    )
    args = parser.parse_args()

    out_root = Path(args.out)
    staging: Path | None = None

    if args.zip:
        zip_path = Path(args.zip)
        if not zip_path.is_file():
            sys.exit(f"Zip not found: {zip_path}")
        staging = Path("data_raw")
        if staging.exists():
            shutil.rmtree(staging)
        search_root = extract_zip(zip_path, staging)
    else:
        search_root = Path(args.src)
        if not search_root.is_dir():
            sys.exit(f"Source folder not found: {search_root}")

    splits = find_split_dirs(search_root)
    if "train" not in splits:
        sys.exit(
            f"Could not find a 'train' directory with class subfolders under {search_root}.\n"
            "Inspect the archive and pass the correct --src path."
        )
    print(f"Found train: {splits['train']}")
    if "test" in splits:
        print(f"Found test:  {splits['test']}")
    else:
        print("WARNING: no test directory found. Training will still work; test scoring will not.")

    train_counts = copy_split(splits["train"], out_root / "train")
    test_counts = copy_split(splits["test"], out_root / "test") if "test" in splits else Counter()

    print(f"\nClasses: {len(train_counts)}")
    print(f"{'class':<28}{'train':>8}{'test':>8}")
    for class_name in sorted(train_counts):
        print(f"{class_name:<28}{train_counts[class_name]:>8}{test_counts.get(class_name, 0):>8}")
    print(f"{'TOTAL':<28}{sum(train_counts.values()):>8}{sum(test_counts.values()):>8}")

    if test_counts and set(test_counts) != set(train_counts):
        print(
            "\nWARNING: train and test class names differ:"
            f"\n  train-only: {sorted(set(train_counts) - set(test_counts))}"
            f"\n  test-only:  {sorted(set(test_counts) - set(train_counts))}"
        )

    expected_total = 2400
    actual_total = sum(train_counts.values())
    if actual_total != expected_total:
        print(
            f"\nNOTE: expected {expected_total} training images per the assignment, found {actual_total}. "
            "If this is lower, the download may be incomplete."
        )

    if staging is not None and not args.keep_staging:
        shutil.rmtree(staging, ignore_errors=True)
        print("\nRemoved staging folder data_raw/")

    print(f"\nDataset ready at {out_root.resolve()}")


if __name__ == "__main__":
    main()
