"""Seeding, device selection, metric tracking and run bookkeeping."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and Torch.

    `deterministic=True` also pins cuDNN algorithm selection. It is off by
    default because it measurably slows convolutions, and run-to-run variance is
    better characterised by repeating a run with different seeds than by forcing
    bitwise determinism.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """DataLoader worker seeding so augmentation streams are reproducible."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter:
    """Running average of a scalar, weighted by batch size."""

    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


def environment_info() -> dict[str, Any]:
    """Capture the software/hardware context needed to interpret a result."""
    info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    try:
        import torchvision

        info["torchvision"] = torchvision.__version__
    except ImportError:
        info["torchvision"] = None
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    try:
        info["git_commit"] = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        info["git_commit"] = None
    return info


class RunDirectory:
    """Owns the on-disk artifacts of a single training run."""

    def __init__(self, root: str | Path, name: str) -> None:
        self.path = Path(root) / name
        self.path.mkdir(parents=True, exist_ok=True)
        self.history_path = self.path / "history.jsonl"
        self.best_checkpoint = self.path / "best.pt"
        self.last_checkpoint = self.path / "last.pt"
        self.summary_path = self.path / "summary.json"

    def log_epoch(self, record: dict[str, Any]) -> None:
        with open(self.history_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def read_history(self) -> list[dict[str, Any]]:
        if not self.history_path.exists():
            return []
        with open(self.history_path, "r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def write_summary(self, summary: dict[str, Any]) -> None:
        with open(self.summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    cfg: dict[str, Any],
    class_names: list[str],
    extra: dict[str, Any] | None = None,
) -> None:
    """Save weights together with everything needed to rebuild the model."""
    payload = {
        "model_state": model.state_dict(),
        "config": cfg,
        "class_names": class_names,
        "environment": environment_info(),
    }
    if extra:
        payload.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    # weights_only=False because we deliberately store the config dict alongside
    # the tensors; only load checkpoints you produced yourself.
    return torch.load(path, map_location=map_location, weights_only=False)
