"""Unpack the downloaded dataset into the layout the training code expects.

    python scripts/prepare_data.py --zip "C:/Users/you/Downloads/dataset.zip"
    python scripts/prepare_data.py --src "C:/Users/you/Downloads/extracted_folder"

Target layout:

    data/
      train/<class_name>/*.jpg
      test/<class_name>/*.jpg

The script discovers every split directory wherever it sits inside the source
tree, rather than assuming a fixed depth or a fixed set of names. The provided
dataset ships `train`, `test` and an additional `test2`, so all splits found are
copied through under their own names and reported; which one is scored is then a
config choice (`data.test_dir`) instead of something hardcoded here.

macOS metadata entries are skipped, class names are cross-checked between splits,
and per-class counts are printed so a truncated download is obvious before any
GPU time is spent.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Names that mean "this is the training split". Everything else that looks like a
# split is carried through under its own name.
TRAIN_ALIASES = {"train", "training", "train_set"}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTENSIONS and not path.name.startswith(".")


def is_class_dir(path: Path) -> bool:
    """A class directory holds images *directly*, not in nested subfolders."""
    if not path.is_dir() or path.name.startswith("."):
        return False
    return any(is_image(child) for child in path.iterdir() if child.is_file())


def is_split_dir(path: Path) -> bool:
    """A split directory's children are class directories.

    Requiring images one level down (not merely somewhere below) is what stops
    the dataset root itself from being mistaken for a split: `data/` contains
    `train/`, which contains images only two levels down.
    """
    if not path.is_dir() or path.name.startswith("."):
        return False
    class_dirs = [child for child in path.iterdir() if is_class_dir(child)]
    return len(class_dirs) >= 2


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
    """Find every split directory in the tree, keyed by its own folder name.

    The training split is normalised to the key `train` so the rest of the code
    can rely on it; other splits keep their source names (`test`, `test2`, ...).
    """
    found: dict[str, Path] = {}
    for candidate in [root, *sorted(p for p in root.rglob("*") if p.is_dir())]:
        if not is_split_dir(candidate):
            continue
        name = candidate.name.lower()
        key = "train" if name in TRAIN_ALIASES else name
        found.setdefault(key, candidate)
    return found


def find_flat_dirs(root: Path, split_dirs: set[Path]) -> dict[str, Path]:
    """Find unlabelled image directories: loose images, no class subfolders.

    This dataset's `test2` is 400 files named `image_0.jpg`..`image_399.jpg` with no
    class structure, so it is a prediction set rather than a scoreable split. It is
    staged locally alongside the real splits so `predict.py` does not have to read
    across the Drive mount.
    """
    found: dict[str, Path] = {}
    for candidate in sorted(p for p in root.iterdir() if p.is_dir()):
        if candidate in split_dirs or not is_class_dir(candidate):
            continue
        # A split's class folders are also "class dirs"; only consider top-level
        # directories that are not themselves inside a detected split.
        if any(split_dir in candidate.parents for split_dir in split_dirs):
            continue
        found[candidate.name.lower()] = candidate
    return found


def copy_flat(src: Path, dest: Path) -> int:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    for image in sorted(src.iterdir()):
        if image.is_file() and is_image(image):
            shutil.copy2(image, dest / image.name)
            count += 1
    return count


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
            f"Splits detected: {sorted(splits) or 'none'}\n"
            "Inspect the source and pass the correct --src path."
        )
    for name in sorted(splits):
        print(f"Found split '{name}': {splits[name]}")
    if len(splits) == 1:
        print("WARNING: only a training split was found; test scoring will not be possible.")

    # Copy train first so it can anchor the class-name comparison.
    order = ["train"] + sorted(k for k in splits if k != "train")
    counts: dict[str, Counter] = {}
    for name in order:
        print(f"Copying '{name}' ...", flush=True)
        counts[name] = copy_split(splits[name], out_root / name)

    train_counts = counts["train"]
    other_names = [n for n in order if n != "train"]

    print(f"\nClasses: {len(train_counts)}")
    header = f"{'class':<28}" + "".join(f"{n:>8}" for n in order)
    print(header)
    for class_name in sorted(train_counts):
        row = f"{class_name:<28}" + "".join(f"{counts[n].get(class_name, 0):>8}" for n in order)
        print(row)
    print(f"{'TOTAL':<28}" + "".join(f"{sum(counts[n].values()):>8}" for n in order))

    for name in other_names:
        if set(counts[name]) != set(train_counts):
            print(
                f"\nWARNING: class names in '{name}' differ from 'train':"
                f"\n  train-only: {sorted(set(train_counts) - set(counts[name]))}"
                f"\n  {name}-only: {sorted(set(counts[name]) - set(train_counts))}"
                "\nImageFolder assigns label indices alphabetically per directory, so a "
                "mismatched class list would silently mislabel predictions. This split "
                "cannot be used for scoring as-is."
            )

    expected_total = 2400
    actual_total = sum(train_counts.values())
    if actual_total != expected_total:
        print(
            f"\nNOTE: the assignment specifies {expected_total} training images, found {actual_total}. "
            "A lower count means the copy or download is incomplete."
        )

    if len(other_names) > 1:
        print(
            f"\nNOTE: multiple non-training splits present ({', '.join(other_names)}). "
            "The default config scores 'test'; select another with "
            "--set data.test_dir=<name>. Decide which split is the graded one before "
            "reporting a final number."
        )

    # Unlabelled prediction directories (this dataset's test2).
    flat_dirs = find_flat_dirs(search_root, set(splits.values()))
    for name, src in flat_dirs.items():
        n_copied = copy_flat(src, out_root / name)
        print(
            f"\nUnlabelled image directory '{name}': {n_copied} images copied "
            f"(no class subfolders, so it cannot be scored).\n"
            f"  Generate a submission with:\n"
            f"    python predict.py --checkpoint <ckpt> --images-dir {out_root / name} "
            f"--out reports/{name}_predictions.csv\n"
            f"  Check whether it duplicates a labelled split with:\n"
            f"    python scripts/check_split_overlap.py --labelled {out_root / 'test'} "
            f"--flat {out_root / name}"
        )

    if staging is not None and not args.keep_staging:
        shutil.rmtree(staging, ignore_errors=True)
        print("\nRemoved staging folder data_raw/")

    print(f"\nDataset ready at {out_root.resolve()}")


if __name__ == "__main__":
    main()
