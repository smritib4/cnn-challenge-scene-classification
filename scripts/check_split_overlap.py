"""Check whether two splits contain the same image files, by content hash.

    python scripts/check_split_overlap.py --labelled data/test --flat data_raw/test2

Why this matters: the dataset ships `test` (400 labelled images in class folders)
and `test2` (400 loose unlabelled images). If they are the same pictures, then the
accuracy measured on `test` already *is* the accuracy on `test2`, and no separate
claim is needed. If they are different pictures, `test2` is a genuinely unseen set
and nothing about its accuracy can be inferred from `test`.

Guessing either way would be sloppy, and the check costs seconds. Comparison is by
SHA-256 of file bytes, so renaming is irrelevant. Note that re-encoding the same
picture would defeat a byte hash, so a reported overlap of zero means "not
byte-identical", not necessarily "visually different".
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_labelled(root: Path) -> dict[str, tuple[str, str]]:
    """Map content hash -> (class_name, filename) for a class-folder split."""
    mapping: dict[str, tuple[str, str]] = {}
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for image in sorted(class_dir.iterdir()):
            if image.is_file() and image.suffix.lower() in IMG_EXTENSIONS:
                mapping[file_hash(image)] = (class_dir.name, image.name)
    return mapping


def hash_flat(root: Path) -> dict[str, str]:
    """Map filename -> content hash for a flat directory of images."""
    return {
        image.name: file_hash(image)
        for image in sorted(root.iterdir())
        if image.is_file() and image.suffix.lower() in IMG_EXTENSIONS
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labelled", required=True, help="Split with class subfolders, e.g. data/test")
    parser.add_argument("--flat", required=True, help="Flat directory of images, e.g. data_raw/test2")
    parser.add_argument(
        "--out",
        default=None,
        help="If the sets match, write a filename,class CSV of recovered labels here",
    )
    args = parser.parse_args()

    labelled_root = Path(args.labelled)
    flat_root = Path(args.flat)
    print(f"Hashing labelled split: {labelled_root}")
    labelled = hash_labelled(labelled_root)
    print(f"Hashing flat split:     {flat_root}")
    flat = hash_flat(flat_root)

    print(f"\nlabelled images: {len(labelled)}")
    print(f"flat images:     {len(flat)}")

    matched = {name: labelled[h] for name, h in flat.items() if h in labelled}
    print(f"byte-identical matches: {len(matched)} / {len(flat)}")

    if not matched:
        print(
            "\nCONCLUSION: no byte-identical overlap. Treat the flat split as a separate,\n"
            "unseen set: its accuracy cannot be inferred from the labelled split, and\n"
            "labels for it are unknown (use predict.py to produce a submission file)."
        )
        return

    if len(matched) == len(flat) == len(labelled):
        print(
            "\nCONCLUSION: the two splits are the same images, only renamed and flattened.\n"
            "Accuracy measured on the labelled split therefore already applies to the flat\n"
            "split, and the labels below are exact rather than predicted."
        )
    else:
        print(
            f"\nCONCLUSION: partial overlap ({len(matched)} of {len(flat)}). The splits are "
            "related but not identical; report them separately."
        )

    distribution = Counter(class_name for class_name, _ in matched.values())
    print("\nRecovered class distribution:")
    for class_name in sorted(distribution):
        print(f"  {class_name:<16} {distribution[class_name]:>4}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["filename", "class", "source_filename"])
            for name in sorted(matched):
                class_name, source_name = matched[name]
                writer.writerow([name, class_name, source_name])
        print(f"\nWrote recovered labels to {out_path}")


if __name__ == "__main__":
    main()
