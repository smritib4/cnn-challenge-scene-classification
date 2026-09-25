"""Render the experiment table straight from the recorded run results.

    python scripts/make_report_table.py
    python scripts/make_report_table.py --seed-summary

Every `train.py` run appends a row to `experiments/results/results.csv`. Generating
the report table from that file rather than retyping numbers means the table cannot
drift from what was actually run, and a claimed result always has a run directory
behind it.

`--seed-summary` groups repeated runs of the same experiment and reports mean and
spread across seeds, which is what says whether a 1% difference is real: the
validation set is ~480 images, so 1% is about five images.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

RESULTS_CSV = Path("experiments/results/results.csv")


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"No results file at {path}. Run train.py at least once.")
    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt_pct(value: str) -> str:
    if value in ("", None):
        return "-"
    return f"{float(value) * 100:.2f}"


def fmt_params(value: str) -> str:
    if not value:
        return "-"
    n = int(value)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def fmt_time(value: str) -> str:
    if not value:
        return "-"
    seconds = float(value)
    if seconds < 90:
        return f"{seconds:.0f}s"
    return f"{seconds / 60:.0f}m"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=str(RESULTS_CSV))
    parser.add_argument(
        "--seed-summary",
        action="store_true",
        help="Aggregate repeated runs of the same experiment across seeds",
    )
    args = parser.parse_args()

    rows = load_rows(Path(args.csv))
    rows.sort(key=lambda r: r["experiment"])

    if args.seed_summary:
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            if row.get("val_acc"):
                grouped[row["experiment"]].append(float(row["val_acc"]))
        print("| Experiment | Seeds | Mean val acc (%) | Spread (pp) |")
        print("|---|---:|---:|---:|")
        for name in sorted(grouped):
            values = grouped[name]
            mean = statistics.mean(values) * 100
            spread = (max(values) - min(values)) * 100 if len(values) > 1 else 0.0
            stdev = statistics.stdev(values) * 100 if len(values) > 2 else None
            spread_text = f"{spread:.2f}" + (f" (sd {stdev:.2f})" if stdev is not None else "")
            print(f"| {name} | {len(values)} | {mean:.2f} | {spread_text} |")
        return

    print("| Experiment | Model | Res | Aug | Epochs | Params | Best ep | Val acc (%) | Test acc (%) | Time |")
    print("|---|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['experiment']} "
            f"| {row['model']}{'' if row.get('pretrained') == 'True' else ' (scratch)'} "
            f"| {row['img_size']} "
            f"| {row['augment']} "
            f"| {row['epochs']} "
            f"| {fmt_params(row.get('params', ''))} "
            f"| {row.get('best_epoch', '-')} "
            f"| {fmt_pct(row.get('val_acc', ''))} "
            f"| {fmt_pct(row.get('test_acc', ''))} "
            f"| {fmt_time(row.get('train_time_s', ''))} |"
        )

    scored = [r for r in rows if r.get("val_acc")]
    if scored:
        best = max(scored, key=lambda r: float(r["val_acc"]))
        print(
            f"\nBest by validation accuracy: {best['experiment']} "
            f"({fmt_pct(best['val_acc'])}%), run dir {best.get('run_dir', '?')}"
        )


if __name__ == "__main__":
    main()
