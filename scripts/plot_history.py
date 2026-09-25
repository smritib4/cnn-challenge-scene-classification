"""Plot training curves from one or more runs.

    python scripts/plot_history.py --runs runs/05_resnet50_finetune_seed0 \
        --out reports/overfitting_resnet50.png

    python scripts/plot_history.py --runs runs/05_* runs/06_* --out reports/aug_effect.png

The left panel shows train and validation loss, the right panel validation
accuracy. The train/validation loss gap is the evidence for or against
overfitting, which is the diagnosis that drives the augmentation experiments -
"the model is overfitting" is an assertion until this gap is shown.

Comparing two runs on the same axes is what makes an ablation legible: if strong
augmentation helped, the gap narrows and the accuracy curve rises later but
higher.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_history(run_dir: Path) -> list[dict]:
    path = run_dir / "history.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"No history.jsonl in {run_dir}")
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def selected_accuracy(record: dict) -> float:
    """Validation accuracy of whichever weights would be selected this epoch.

    When EMA is enabled, `fit()` scores both the live model and its EMA copy and
    keeps the better one, so reading `val_acc` alone understates the accuracy of
    the checkpoint actually saved.
    """
    return max(float(record["val_acc"]), float(record.get("val_acc_ema", 0.0)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories to plot")
    parser.add_argument("--out", required=True, help="Output PNG path")
    parser.add_argument("--labels", nargs="*", default=None, help="Optional legend labels")
    args = parser.parse_args()

    run_dirs = [Path(r) for r in args.runs]
    labels = args.labels or [d.name.replace("_seed0", "") for d in run_dirs]
    if len(labels) != len(run_dirs):
        raise SystemExit(f"Got {len(run_dirs)} runs but {len(labels)} labels")

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 4.5))
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for index, (run_dir, label) in enumerate(zip(run_dirs, labels)):
        history = read_history(run_dir)
        epochs = [record["epoch"] for record in history]
        colour = colours[index % len(colours)]

        ax_loss.plot(epochs, [r["train_loss"] for r in history], color=colour, label=f"{label} train")
        ax_loss.plot(
            epochs,
            [r["val_loss"] for r in history],
            color=colour,
            linestyle="--",
            label=f"{label} val",
        )
        ax_acc.plot(epochs, [selected_accuracy(r) for r in history], color=colour, label=label)
        if any("val_acc_ema" in r for r in history):
            ax_acc.plot(
                epochs,
                [r["val_acc"] for r in history],
                color=colour,
                linestyle=":",
                alpha=0.6,
                label=f"{label} (no EMA)",
            )

        best = max(history, key=selected_accuracy)
        # Mark the selected checkpoint: model selection is on validation accuracy,
        # so the reported number is this point, not the final epoch.
        ax_acc.scatter([best["epoch"]], [selected_accuracy(best)], color=colour, zorder=5, s=40)

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss (solid = train, dashed = validation)")
    ax_loss.legend(fontsize=8)
    ax_loss.grid(alpha=0.3)

    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Validation accuracy")
    ax_acc.set_title("Validation accuracy (dot = selected checkpoint)")
    ax_acc.legend(fontsize=8)
    ax_acc.grid(alpha=0.3)

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")

    for run_dir, label in zip(run_dirs, labels):
        history = read_history(run_dir)
        best = max(history, key=selected_accuracy)
        final = history[-1]
        gap = final["val_loss"] - final["train_loss"]
        print(
            f"{label:<28} best val acc {selected_accuracy(best):.4f} @ epoch {best['epoch']:>3} | "
            f"final train/val loss {final['train_loss']:.3f}/{final['val_loss']:.3f} "
            f"(gap {gap:+.3f})"
        )


if __name__ == "__main__":
    main()
