"""Rank every run by its best validation epoch, including runs that never finished.

    python scripts/pick_best_checkpoint.py            # list only
    python scripts/pick_best_checkpoint.py --evaluate  # also score the test split

`results.csv` only gets a row when train.py reaches the end, so a run killed part
way through is invisible there even though its epoch history and its best-epoch
checkpoint are both on disk. This reads history.jsonl directly so the selection
sees those runs too.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DRIVE = Path("/content/gdrive/MyDrive/a1_artifacts")


def best_epoch(history: Path) -> tuple[float, int, int]:
    """Best validation accuracy in a run, the epoch it happened, and epochs seen."""
    best, best_ep, seen = 0.0, 0, 0
    for line in history.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        seen = int(rec["epoch"])
        # EMA weights are a separate candidate; train.py selects whichever is higher.
        acc = max(float(rec["val_acc"]), float(rec.get("val_acc_ema", 0.0)))
        if acc > best:
            best, best_ep = acc, seen
    return best, best_ep, seen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evaluate", action="store_true", help="Score the winner on the test split")
    parser.add_argument("--runs-root", default="runs")
    args = parser.parse_args()

    rows = []
    for run_dir in sorted((REPO / args.runs_root).glob("*_seed*")):
        history = run_dir / "history.jsonl"
        if not history.exists():
            continue
        acc, ep, seen = best_epoch(history)
        checkpoint = run_dir / "best.pt"
        rows.append(
            {
                "run": run_dir.name,
                "val_acc": acc,
                "best_epoch": ep,
                "epochs_seen": seen,
                "finished": (run_dir / "summary.json").exists(),
                "checkpoint": checkpoint if checkpoint.exists() else None,
            }
        )

    rows.sort(key=lambda r: r["val_acc"], reverse=True)
    print(f"{'run':<32}{'val_acc':>9}{'@ep':>5}{'epochs':>8}{'done':>6}  checkpoint")
    for r in rows:
        print(
            f"{r['run']:<32}{r['val_acc'] * 100:>8.2f}%{r['best_epoch']:>5}"
            f"{r['epochs_seen']:>8}{str(r['finished']):>6}  "
            f"{'yes' if r['checkpoint'] else 'MISSING'}"
        )

    usable = [r for r in rows if r["checkpoint"]]
    if not usable:
        sys.exit("\nNo checkpoints on disk.")
    win = usable[0]
    print(f"\nBest validation accuracy with a checkpoint on disk: {win['run']} at {win['val_acc'] * 100:.2f}%")
    if not win["finished"]:
        print("NOTE: this run did not finish. The checkpoint is still its best validation epoch.")

    if DRIVE.parent.exists():
        DRIVE.mkdir(parents=True, exist_ok=True)
        target = DRIVE / f"{win['run']}_best.pt"
        target.write_bytes(win["checkpoint"].read_bytes())
        print(f"Backed up to {target} ({target.stat().st_size / 1e6:.1f} MB)")

    if not args.evaluate:
        print("\nRe-run with --evaluate to score the test split.")
        return

    subprocess.run(
        [
            sys.executable, "evaluate.py",
            "--checkpoint", str(win["checkpoint"]),
            "--split", "test",
            "--test-dir", "test",
            "--batch-size", "32",
            "--report", f"reports/test_report_{win['run']}.json",
            "--confusion-matrix", f"reports/confusion_matrix_{win['run']}.png",
        ],
        cwd=REPO,
        check=True,
    )


if __name__ == "__main__":
    main()
