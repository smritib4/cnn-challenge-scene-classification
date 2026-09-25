"""Generate a tiny synthetic dataset with the real directory layout.

Used to smoke-test the training pipeline without the real images, and to keep a
runnable end-to-end check in the repo:

    python scripts/make_dummy_data.py --out data_dummy
    python train.py --config configs/smoke.yaml --data-root data_dummy

Each class gets a distinct base hue plus noise, so a model should be able to fit
it. That makes the check meaningful: if training loss does not fall on this data,
the bug is in the pipeline, not the dataset.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

CLASS_NAMES = [
    "bedroom", "coast", "forest", "highway", "industrial", "inside_city",
    "kitchen", "living_room", "mountain", "office", "open_country", "store",
    "street", "suburb", "tall_building", "warehouse",
]


def make_split(root: Path, per_class: int, size: int, seed: int) -> int:
    rng = np.random.default_rng(seed)
    total = 0
    for class_id, name in enumerate(CLASS_NAMES):
        class_dir = root / name
        class_dir.mkdir(parents=True, exist_ok=True)
        # Distinct hue per class so the task is learnable rather than pure noise.
        base = np.array(
            [
                (class_id * 37) % 256,
                (class_id * 91 + 40) % 256,
                (class_id * 53 + 90) % 256,
            ],
            dtype=np.float32,
        )
        for i in range(per_class):
            noise = rng.normal(0, 25, size=(size, size, 3))
            array = np.clip(base[None, None, :] + noise, 0, 255).astype(np.uint8)
            Image.fromarray(array).save(class_dir / f"{name}_{i:04d}.jpg", quality=90)
            total += 1
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=str, default="data_dummy")
    parser.add_argument("--train-per-class", type=int, default=12)
    parser.add_argument("--test-per-class", type=int, default=4)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    n_train = make_split(out / "train", args.train_per_class, args.size, args.seed)
    n_test = make_split(out / "test", args.test_per_class, args.size, args.seed + 1)
    print(f"Wrote {n_train} train and {n_test} test images across {len(CLASS_NAMES)} classes to {out}")


if __name__ == "__main__":
    main()
