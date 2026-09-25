"""Train a CNN on the 16-class scene dataset.

    python train.py --config configs/best.yaml

Single-factor ablations are run by overriding one key, which keeps the rest of
the recipe identical:

    python train.py --config configs/best.yaml --set augment.preset=basic
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch  # noqa: E402

from cnn_challenge.config import load_config, save_config  # noqa: E402
from cnn_challenge.data import build_datasets, build_loaders, describe_dataset  # noqa: E402
from cnn_challenge.engine import evaluate, fit  # noqa: E402
from cnn_challenge.models import build_model  # noqa: E402
from cnn_challenge.utils import (  # noqa: E402
    RunDirectory,
    count_parameters,
    environment_info,
    get_device,
    save_checkpoint,
    set_seed,
)

RESULTS_CSV = Path("experiments/results/results.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML experiment config")
    parser.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=[],
        metavar="key.subkey=value",
        help="Override config values, e.g. --set optim.lr=1e-4 model.name=resnet18",
    )
    parser.add_argument("--name", type=str, default=None, help="Override the run name")
    parser.add_argument("--seed", type=int, default=None, help="Override the random seed")
    parser.add_argument("--data-root", type=str, default=None, help="Override data.root")
    parser.add_argument("--device", type=str, default=None, help="cuda, cpu or mps")
    parser.add_argument("--output-root", type=str, default=None, help="Override output.root")
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Also score the held-out test set. Use only for a final, already-selected model.",
    )
    parser.add_argument("--deterministic", action="store_true", help="Force deterministic cuDNN kernels")
    return parser.parse_args()


def append_result_row(row: dict[str, object]) -> None:
    """Append one line per run to a CSV so the experiment table stays honest."""
    import csv

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    if args.name:
        cfg["experiment"]["name"] = args.name
    if args.seed is not None:
        cfg["experiment"]["seed"] = args.seed
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    if args.output_root:
        cfg["output"]["root"] = args.output_root

    set_seed(int(cfg["experiment"]["seed"]), deterministic=args.deterministic)
    device = get_device(args.device)

    env = environment_info()
    print(json.dumps(env, indent=2))
    print(f"Device: {device}")

    bundle = build_datasets(cfg)
    loaders = build_loaders(cfg, bundle)
    print(describe_dataset(bundle))

    model = build_model(cfg, num_classes=len(bundle["class_names"]))
    n_params = count_parameters(model)
    print(f"Model: {cfg['model']['name']} | trainable parameters: {n_params:,}")

    run_name = f"{cfg['experiment']['name']}_seed{cfg['experiment']['seed']}"
    run = RunDirectory(cfg["output"]["root"], run_name)
    save_config(cfg, run.path / "config.yaml")

    result = fit(model, loaders, cfg, device, on_epoch_end=run.log_epoch)
    model = result["model"]

    val_stats = evaluate(
        model,
        loaders["val"],
        device,
        tta=str(cfg["eval"].get("tta", "none")),
    )
    print(f"Selected model | val acc {val_stats['acc']:.4f} | val top-5 {val_stats['top5_acc']:.4f}")

    summary = {
        "experiment": cfg["experiment"]["name"],
        "seed": cfg["experiment"]["seed"],
        "model": cfg["model"]["name"],
        "pretrained": cfg["model"]["pretrained"],
        "img_size": cfg["data"]["img_size"],
        "augment_preset": cfg["augment"]["preset"],
        "epochs": cfg["optim"]["epochs"],
        "trainable_params": n_params,
        "best_epoch": result["best_epoch"],
        "best_used_ema": result["best_used_ema"],
        "val_acc": round(val_stats["acc"], 4),
        "val_top5": round(val_stats["top5_acc"], 4),
        "val_loss": round(val_stats["loss"], 4),
        "train_time_s": result["train_time_s"],
        "environment": env,
    }

    if cfg["output"].get("save_best", True):
        save_checkpoint(
            run.best_checkpoint,
            model,
            cfg,
            bundle["class_names"],
            extra={"val_acc": val_stats["acc"], "best_epoch": result["best_epoch"]},
        )
        summary["checkpoint"] = str(run.best_checkpoint)
        print(f"Saved checkpoint: {run.best_checkpoint}")

    if args.evaluate_test:
        if loaders["test"] is None:
            print("No test directory found; skipping test evaluation.")
        else:
            test_stats = evaluate(
                model, loaders["test"], device, tta=str(cfg["eval"].get("tta", "none"))
            )
            summary["test_acc"] = round(test_stats["acc"], 4)
            summary["test_top5"] = round(test_stats["top5_acc"], 4)
            print(f"TEST accuracy: {test_stats['acc']:.4f} | top-5 {test_stats['top5_acc']:.4f}")

    run.write_summary(summary)
    append_result_row(
        {
            "experiment": summary["experiment"],
            "seed": summary["seed"],
            "model": summary["model"],
            "pretrained": summary["pretrained"],
            "img_size": summary["img_size"],
            "augment": summary["augment_preset"],
            "epochs": summary["epochs"],
            "params": summary["trainable_params"],
            "best_epoch": summary["best_epoch"],
            "val_acc": summary["val_acc"],
            "test_acc": summary.get("test_acc", ""),
            "train_time_s": summary["train_time_s"],
            "run_dir": str(run.path),
        }
    )
    print(f"Summary written to {run.summary_path}")


if __name__ == "__main__":
    main()
