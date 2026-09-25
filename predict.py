"""Predict labels for a directory of unlabelled images.

    python predict.py --checkpoint runs/best_seed0/best.pt \
        --images-dir data_raw/test2 --out reports/test2_predictions.csv

The dataset ships a `test2` directory holding 400 loose, unlabelled JPEGs
(`image_0.jpg` ... `image_399.jpg`) rather than class subdirectories, so
`ImageFolder` cannot read it and `evaluate.py` does not apply — there are no
labels to score against. This script writes a predictions CSV instead.

The model, class ordering and evaluation transform all come from the checkpoint,
so predictions cannot silently drift from the training-time preprocessing.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from cnn_challenge.config import apply_overrides  # noqa: E402
from cnn_challenge.data import FlatImageDataset  # noqa: E402
from cnn_challenge.models import build_model  # noqa: E402
from cnn_challenge.transforms import build_transforms  # noqa: E402
from cnn_challenge.utils import get_device, load_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--images-dir", type=str, required=True, help="Directory of unlabelled images"
    )
    parser.add_argument("--out", type=str, default="reports/predictions.csv")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tta", choices=["none", "hflip"], default=None, help="Override eval.tta")
    parser.add_argument(
        "--all-probs",
        action="store_true",
        help="Also write one probability column per class",
    )
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    payload = load_checkpoint(args.checkpoint)
    cfg = payload["config"]
    class_names = payload["class_names"]
    if args.tta:
        cfg = apply_overrides(cfg, [f"eval.tta={args.tta}"])
    tta = str(cfg["eval"].get("tta", "none"))

    device = get_device(args.device)
    # Same transform the checkpoint was validated with.
    transform = build_transforms(cfg["data"], cfg["augment"], train=False)
    dataset = FlatImageDataset(args.images_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(cfg, num_classes=len(class_names))
    model.load_state_dict(payload["model_state"])
    model = model.to(device).eval()

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Images:     {len(dataset)} from {args.images_dir}")
    print(f"Resolution: {cfg['data']['img_size']} | TTA: {tta}")

    rows: list[dict[str, object]] = []
    for images, names in loader:
        images = images.to(device, non_blocking=True)
        probs = F.softmax(model(images), dim=1)
        if tta == "hflip":
            probs = 0.5 * (probs + F.softmax(model(torch.flip(images, dims=[3])), dim=1))
        confidences, indices = probs.max(dim=1)
        for i, name in enumerate(names):
            row: dict[str, object] = {
                "filename": name,
                "predicted_class": class_names[int(indices[i])],
                "predicted_index": int(indices[i]),
                "confidence": round(float(confidences[i]), 4),
            }
            if args.all_probs:
                for class_index, class_name in enumerate(class_names):
                    row[f"p_{class_name}"] = round(float(probs[i, class_index]), 5)
            rows.append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} predictions to {out_path}")

    # Predicted class balance is a cheap sanity check: this set is 400 images and
    # the labelled test split is 25 per class, so a wildly skewed distribution
    # points at a preprocessing mismatch rather than a genuinely hard set.
    from collections import Counter

    counts = Counter(str(r["predicted_class"]) for r in rows)
    mean_confidence = sum(float(r["confidence"]) for r in rows) / len(rows)
    print(f"Mean confidence: {mean_confidence:.4f}")
    print("Predicted class distribution:")
    for class_name in class_names:
        print(f"  {class_name:<16} {counts.get(class_name, 0):>4}")


if __name__ == "__main__":
    main()
