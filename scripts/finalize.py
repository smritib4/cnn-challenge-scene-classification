"""Produce the submission artifacts from whatever runs have finished.

    python scripts/finalize.py

Selects the best *completed* run by validation accuracy, copies its checkpoint to
Drive, scores the test split exactly once, regenerates the report tables, and
pushes the results. Safe to run while another training job holds the GPU.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "experiments" / "results" / "results.csv"
REPORTS = REPO / "reports"
DRIVE = Path("/content/gdrive/MyDrive/a1_artifacts")


def load_rows() -> list[dict[str, str]]:
    with RESULTS.open(newline="") as fh:
        return list(csv.DictReader(fh))


def drop_smoke_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Pipeline smoke tests run on synthetic data; their accuracy is meaningless."""
    kept = [r for r in rows if r["experiment"] != "smoke"]
    if len(kept) != len(rows):
        with RESULTS.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(kept)
        print(f"Removed {len(rows) - len(kept)} smoke row(s) from {RESULTS.name}")
    return kept


def select_run(rows: list[dict[str, str]]) -> dict[str, str]:
    complete = [r for r in rows if str(r.get("interrupted", "")).lower() != "true"]
    if not complete:
        sys.exit("No completed runs recorded yet.")
    complete.sort(key=lambda r: float(r["val_acc"]), reverse=True)

    print(f"{'experiment':<28} {'seed':>4} {'params':>12} {'epochs':>7} {'val_acc':>8}")
    for row in complete:
        print(
            f"{row['experiment']:<28} {row['seed']:>4} {int(row['params']):>12,} "
            f"{row.get('epochs_completed', '?'):>7} {float(row['val_acc']) * 100:>7.2f}%"
        )
    return complete[0]


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=REPO, check=True)


def main() -> None:
    rows = drop_smoke_rows(load_rows())
    best = select_run(rows)
    checkpoint = REPO / "runs" / f"{best['experiment']}_seed{best['seed']}" / "best.pt"
    print(f"\nSelected on validation accuracy only: {checkpoint}")
    if not checkpoint.exists():
        sys.exit(f"Checkpoint missing: {checkpoint}")

    # Copy to Drive first: /content is wiped when the runtime dies.
    if DRIVE.parent.exists():
        DRIVE.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint, DRIVE / "final_model.pt")
        size_mb = (DRIVE / "final_model.pt").stat().st_size / 1e6
        print(f"Backed up to {DRIVE / 'final_model.pt'} ({size_mb:.1f} MB)")
    else:
        print("Drive not mounted; skipping checkpoint backup.")

    REPORTS.mkdir(exist_ok=True)
    run(
        [
            sys.executable, "evaluate.py",
            "--checkpoint", str(checkpoint),
            "--split", "test",
            "--test-dir", "test",
            "--batch-size", "32",
            "--report", "reports/final_test_report.json",
            "--confusion-matrix", "reports/final_confusion_matrix.png",
        ]
    )

    for args, out in (([], "experiment_table.md"), (["--seed-summary"], "seed_summary.md")):
        table = subprocess.run(
            [sys.executable, "scripts/make_report_table.py", *args],
            cwd=REPO, check=True, capture_output=True, text=True,
        ).stdout
        (REPORTS / out).write_text(table)
        print(f"\nWrote reports/{out}\n{table}")

    identity = [
        "-c", "user.email=smritib4@users.noreply.github.com",
        "-c", "user.name=smritib4",
    ]
    run(["git", *identity, "add", "-A", "experiments", "reports"])
    run(["git", *identity, "commit", "-q", "-m",
         "Add final test evaluation, per-class report and result tables"])
    run(["git", "push", "-q", "origin", "main"])
    print("\nPUSHED OK")


if __name__ == "__main__":
    main()
