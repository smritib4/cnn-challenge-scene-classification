"""Evaluate a saved checkpoint and write a per-class error report.

    python evaluate.py --checkpoint runs/best_seed0/best.pt --split test

The checkpoint stores its own config and class ordering, so the evaluation
transform is rebuilt exactly as it was at training time rather than re-specified
on the command line.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np  # noqa: E402

from cnn_challenge.config import apply_overrides  # noqa: E402
from cnn_challenge.data import build_datasets, build_loaders  # noqa: E402
from cnn_challenge.engine import evaluate as run_eval  # noqa: E402
from cnn_challenge.models import build_model  # noqa: E402
from cnn_challenge.utils import get_device, load_checkpoint, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--tta", choices=["none", "hflip"], default=None, help="Override eval.tta")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--report", type=str, default=None, help="Optional path to write a JSON report"
    )
    parser.add_argument(
        "--confusion-matrix", type=str, default=None, help="Optional path for a confusion-matrix PNG"
    )
    return parser.parse_args()


def per_class_report(
    targets: np.ndarray, preds: np.ndarray, class_names: list[str]
) -> list[dict[str, object]]:
    """Per-class accuracy and the most frequent wrong prediction."""
    rows = []
    for idx, name in enumerate(class_names):
        mask = targets == idx
        support = int(mask.sum())
        if support == 0:
            continue
        class_preds = preds[mask]
        correct = int((class_preds == idx).sum())
        wrong = class_preds[class_preds != idx]
        confused_with = ""
        if wrong.size:
            values, counts = np.unique(wrong, return_counts=True)
            confused_with = class_names[int(values[counts.argmax()])]
        rows.append(
            {
                "class": name,
                "support": support,
                "accuracy": round(correct / support, 4),
                "most_confused_with": confused_with,
            }
        )
    return sorted(rows, key=lambda r: r["accuracy"])


def save_confusion_matrix(
    targets: np.ndarray, preds: np.ndarray, class_names: list[str], path: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(class_names)
    matrix = np.zeros((n, n), dtype=int)
    for t, p in zip(targets, preds):
        matrix[t, p] += 1
    # Row-normalise so classes with different support stay comparable.
    normalised = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(normalised, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(n), class_names, rotation=90, fontsize=8)
    ax.set_yticks(range(n), class_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Row-normalised confusion matrix")
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix written to {path}")


def main() -> None:
    args = parse_args()
    payload = load_checkpoint(args.checkpoint)
    cfg = payload["config"]
    class_names = payload["class_names"]

    overrides = []
    if args.data_root:
        overrides.append(f"data.root={args.data_root}")
    if args.tta:
        overrides.append(f"eval.tta={args.tta}")
    if args.batch_size:
        overrides.append(f"data.batch_size={args.batch_size}")
    cfg = apply_overrides(cfg, overrides)

    set_seed(int(cfg["experiment"]["seed"]))
    device = get_device(args.device)

    bundle = build_datasets(cfg)
    if bundle["class_names"] != class_names:
        raise ValueError(
            "Class ordering in the data directory differs from the checkpoint; "
            "predictions would be mislabelled."
        )
    loaders = build_loaders(cfg, bundle)
    loader = loaders[args.split]
    if loader is None:
        raise FileNotFoundError(f"No data available for split '{args.split}'")

    model = build_model(cfg, num_classes=len(class_names))
    model.load_state_dict(payload["model_state"])
    model = model.to(device)

    stats = run_eval(
        model,
        loader,
        device,
        tta=str(cfg["eval"].get("tta", "none")),
        return_predictions=True,
    )
    targets = stats["targets"].numpy()
    preds = stats["preds"].numpy()

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split} ({len(targets)} images) | TTA: {cfg['eval'].get('tta', 'none')}")
    print(f"Accuracy:  {stats['acc']:.4f}")
    print(f"Top-5:     {stats['top5_acc']:.4f}")
    print(f"Loss:      {stats['loss']:.4f}")

    rows = per_class_report(targets, preds, class_names)
    print("\nPer-class accuracy (worst first):")
    for row in rows:
        print(
            f"  {row['class']:<24} {row['accuracy']:.3f}  (n={row['support']}, "
            f"most confused with: {row['most_confused_with'] or '-'})"
        )

    if args.confusion_matrix:
        save_confusion_matrix(targets, preds, class_names, args.confusion_matrix)

    if args.report:
        report = {
            "checkpoint": args.checkpoint,
            "split": args.split,
            "tta": cfg["eval"].get("tta", "none"),
            "accuracy": round(stats["acc"], 4),
            "top5_accuracy": round(stats["top5_acc"], 4),
            "loss": round(stats["loss"], 4),
            "per_class": rows,
        }
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nReport written to {args.report}")


if __name__ == "__main__":
    main()
