"""Configuration loading.

Experiments are described by small YAML files that are merged onto the defaults
below, so every config file only needs to state what makes that experiment
different. Command-line `--set key.subkey=value` overrides are applied last,
which is what makes single-variable ablations cheap to run.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "experiment": {
        "name": "unnamed",
        "notes": "",
        "seed": 0,
    },
    "data": {
        "root": "data",
        "train_dir": "train",
        "test_dir": "test",
        "val_fraction": 0.20,
        # "stratified" keeps the per-class count in validation identical, which
        # matters a lot when a class only has ~150 images.
        "split": "stratified",
        "batch_size": 32,
        "num_workers": 4,
        "img_size": 224,
        "grayscale": False,
        # ImageNet statistics; overridden to [0.5]/[0.5] for the grayscale baseline.
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "augment": {
        # "none" -> resize only, "basic" -> flip + crop, "strong" -> + RandAugment/erasing
        "preset": "basic",
        "random_resized_crop": True,
        "crop_scale": [0.65, 1.0],
        "hflip": 0.5,
        "rand_augment": {"enabled": False, "num_ops": 2, "magnitude": 9},
        "color_jitter": 0.0,
        "random_erasing": 0.0,
        "mixup_alpha": 0.0,
        "cutmix_alpha": 0.0,
        "mix_prob": 0.5,
    },
    "model": {
        "name": "resnet50",
        "pretrained": True,
        "dropout": 0.0,
        # "none" trains everything; "backbone" freezes all but the classifier head.
        "freeze": "none",
    },
    "optim": {
        "optimizer": "adamw",
        "lr": 3.0e-4,
        # Pretrained features need a gentler LR than the randomly initialised head.
        "head_lr_mult": 10.0,
        "weight_decay": 0.05,
        "momentum": 0.9,
        "nesterov": True,
        "scheduler": "cosine",
        "warmup_epochs": 2,
        "min_lr": 1.0e-6,
        "epochs": 30,
        "grad_clip": 1.0,
        "label_smoothing": 0.1,
        "amp": True,
        "ema": {"enabled": False, "decay": 0.999},
        "early_stopping_patience": 0,
    },
    "eval": {
        "tta": "none",  # "none" or "hflip"
        "monitor": "val_acc",
    },
    "output": {
        "root": "runs",
        "save_best": True,
        "save_last": False,
    },
}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `updates` into `base`, returning a new dict."""
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _coerce(text: str) -> Any:
    """Turn a CLI string into a bool/int/float/list where that is unambiguous."""
    lowered = text.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text


def apply_overrides(cfg: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply `a.b.c=value` strings onto a config, validating that the key exists."""
    if not overrides:
        return cfg
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must look like key.subkey=value")
        dotted, raw_value = item.split("=", 1)
        keys = dotted.strip().split(".")
        node = cfg
        for key in keys[:-1]:
            if key not in node or not isinstance(node[key], dict):
                raise KeyError(f"Unknown config section '{key}' in override '{item}'")
            node = node[key]
        if keys[-1] not in node:
            raise KeyError(
                f"Unknown config key '{dotted}'. Add it to DEFAULTS before overriding it."
            )
        node[keys[-1]] = _coerce(raw_value)
    return cfg


def load_config(path: str | Path | None, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load a YAML experiment config merged onto DEFAULTS, then apply overrides."""
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        with open(path, "r", encoding="utf-8") as handle:
            file_cfg = yaml.safe_load(handle) or {}
        cfg = deep_update(cfg, file_cfg)
        cfg["experiment"].setdefault("config_path", str(path))
        if cfg["experiment"]["name"] == "unnamed":
            cfg["experiment"]["name"] = Path(path).stem
    return apply_overrides(cfg, overrides)


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
