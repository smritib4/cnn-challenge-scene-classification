"""Optimisation, the training/evaluation loops, and model selection."""

from __future__ import annotations

import copy
import math
import time
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .models import classifier_parameter_names
from .utils import AverageMeter


def build_optimizer(model: nn.Module, optim_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    """Build an optimizer with two decisions baked in.

    * Weight decay is not applied to biases or normalisation parameters. Decaying
      them regularises nothing useful and measurably hurts on small datasets.
    * The randomly initialised head gets `head_lr_mult` times the backbone LR.
      The backbone already encodes useful features and only needs gentle
      adjustment, whereas the head starts from noise.
    """
    head_names = classifier_parameter_names(model)
    base_lr = float(optim_cfg["lr"])
    head_lr = base_lr * float(optim_cfg.get("head_lr_mult", 1.0))
    weight_decay = float(optim_cfg["weight_decay"])

    groups: dict[str, dict[str, Any]] = {
        "backbone_decay": {"params": [], "lr": base_lr, "weight_decay": weight_decay},
        "backbone_nodecay": {"params": [], "lr": base_lr, "weight_decay": 0.0},
        "head_decay": {"params": [], "lr": head_lr, "weight_decay": weight_decay},
        "head_nodecay": {"params": [], "lr": head_lr, "weight_decay": 0.0},
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_head = name in head_names
        no_decay = param.ndim <= 1 or name.endswith(".bias")
        key = f"{'head' if is_head else 'backbone'}_{'nodecay' if no_decay else 'decay'}"
        groups[key]["params"].append(param)

    param_groups = [g for g in groups.values() if g["params"]]
    if not param_groups:
        raise ValueError("No trainable parameters; check the model freeze setting.")

    name = str(optim_cfg.get("optimizer", "adamw")).lower()
    if name == "adamw":
        return torch.optim.AdamW(param_groups, lr=base_lr, betas=(0.9, 0.999))
    if name == "adam":
        return torch.optim.Adam(param_groups, lr=base_lr)
    if name == "sgd":
        return torch.optim.SGD(
            param_groups,
            lr=base_lr,
            momentum=float(optim_cfg.get("momentum", 0.9)),
            nesterov=bool(optim_cfg.get("nesterov", True)),
        )
    raise ValueError(f"Unknown optimizer '{name}' (use adamw, adam or sgd)")


def build_scheduler(
    optimizer: torch.optim.Optimizer, optim_cfg: dict[str, Any], steps_per_epoch: int
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Per-step LR schedule with linear warmup.

    Stepping per batch rather than per epoch matters here: with ~60 batches per
    epoch and only ~30 epochs, an epoch-granular cosine is a very coarse curve.
    """
    kind = str(optim_cfg.get("scheduler", "cosine")).lower()
    if kind in {"none", "constant", ""}:
        return None

    epochs = int(optim_cfg["epochs"])
    warmup_steps = int(float(optim_cfg.get("warmup_epochs", 0)) * steps_per_epoch)
    total_steps = max(1, epochs * steps_per_epoch)
    min_lr_ratio = float(optim_cfg.get("min_lr", 0.0)) / max(float(optim_cfg["lr"]), 1e-12)
    min_lr_ratio = min(min_lr_ratio, 1.0)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        if kind == "cosine":
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
        if kind == "step":
            return 0.1 ** int(progress * 3)
        raise ValueError(f"Unknown scheduler '{kind}'")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class ModelEma:
    """Exponential moving average of the weights.

    The EMA weights are usually a little flatter and generalise slightly better
    than the final SGD iterate, which helps when the validation set is only ~480
    images and single-epoch noise is large.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.module = copy.deepcopy(model).eval()
        for param in self.module.parameters():
            param.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_p, model_p in zip(self.module.state_dict().values(), model.state_dict().values()):
            if ema_p.dtype.is_floating_point:
                ema_p.mul_(self.decay).add_(model_p.detach(), alpha=1.0 - self.decay)
            else:
                ema_p.copy_(model_p)


def _rand_bbox(height: int, width: int, lam: float, rng: np.random.Generator) -> tuple[int, int, int, int]:
    """Sample a CutMix box whose area is (1 - lam) of the image."""
    ratio = math.sqrt(1.0 - lam)
    cut_h, cut_w = int(height * ratio), int(width * ratio)
    cy, cx = rng.integers(height), rng.integers(width)
    y1, y2 = np.clip([cy - cut_h // 2, cy + cut_h // 2], 0, height)
    x1, x2 = np.clip([cx - cut_w // 2, cx + cut_w // 2], 0, width)
    return int(y1), int(y2), int(x1), int(x2)


def apply_mix(
    images: torch.Tensor,
    targets: torch.Tensor,
    aug_cfg: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Optionally apply mixup or cutmix.

    Returns `(images, target_a, target_b, lam)`; the caller computes
    `lam * loss(pred, a) + (1 - lam) * loss(pred, b)`. When no mixing happens,
    `lam == 1` and the expression reduces to the plain loss, so the training loop
    needs no special case.
    """
    mixup_alpha = float(aug_cfg.get("mixup_alpha", 0.0) or 0.0)
    cutmix_alpha = float(aug_cfg.get("cutmix_alpha", 0.0) or 0.0)
    mix_prob = float(aug_cfg.get("mix_prob", 0.5) or 0.0)

    if (mixup_alpha <= 0 and cutmix_alpha <= 0) or rng.random() > mix_prob:
        return images, targets, targets, 1.0

    use_cutmix = cutmix_alpha > 0 and (mixup_alpha <= 0 or rng.random() < 0.5)
    alpha = cutmix_alpha if use_cutmix else mixup_alpha
    lam = float(rng.beta(alpha, alpha))
    perm = torch.randperm(images.size(0), device=images.device)

    if use_cutmix:
        y1, y2, x1, x2 = _rand_bbox(images.size(2), images.size(3), lam, rng)
        images = images.clone()
        images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
        # Recompute lam from the actual pixel area, which differs after clipping.
        lam = 1.0 - ((y2 - y1) * (x2 - x1) / (images.size(2) * images.size(3)))
    else:
        images = lam * images + (1.0 - lam) * images[perm]

    return images, targets, targets[perm], lam


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    aug_cfg: dict[str, Any],
    grad_clip: float,
    rng: np.random.Generator,
    ema: ModelEma | None = None,
) -> dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    correct = 0
    total = 0
    amp_enabled = scaler is not None and scaler.is_enabled()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        images, target_a, target_b, lam = apply_mix(images, targets, aug_cfg, rng)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = lam * criterion(logits, target_a) + (1.0 - lam) * criterion(logits, target_b)

        if amp_enabled:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update(model)

        loss_meter.update(loss.item(), images.size(0))
        # With mixing active this "accuracy" is only a rough progress signal;
        # model selection always uses the clean validation set.
        correct += (logits.argmax(1) == targets).sum().item()
        total += targets.size(0)

    return {
        "train_loss": loss_meter.avg,
        "train_acc": correct / max(total, 1),
        "lr": optimizer.param_groups[0]["lr"],
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
    tta: str = "none",
    return_predictions: bool = False,
) -> dict[str, Any]:
    """Evaluate on a clean loader, optionally averaging horizontal-flip TTA.

    Flip TTA is averaged in probability space rather than logit space so the two
    views contribute comparably regardless of their confidence scale.
    """
    model.eval()
    criterion = criterion or nn.CrossEntropyLoss()
    loss_meter = AverageMeter()
    all_probs: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        if tta == "hflip":
            probs = 0.5 * (probs + F.softmax(model(torch.flip(images, dims=[3])), dim=1))
        loss_meter.update(criterion(logits, targets).item(), images.size(0))
        all_probs.append(probs.float().cpu())
        all_targets.append(targets.cpu())

    probs = torch.cat(all_probs)
    targets = torch.cat(all_targets)
    preds = probs.argmax(1)
    top5 = probs.topk(min(5, probs.size(1)), dim=1).indices
    result: dict[str, Any] = {
        "loss": loss_meter.avg,
        "acc": (preds == targets).float().mean().item(),
        "top5_acc": (top5 == targets.unsqueeze(1)).any(dim=1).float().mean().item(),
    }
    if return_predictions:
        result["probs"] = probs
        result["preds"] = preds
        result["targets"] = targets
    return result


def fit(
    model: nn.Module,
    loaders: dict[str, DataLoader | None],
    cfg: dict[str, Any],
    device: torch.device,
    on_epoch_end: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train for `optim.epochs`, keeping the best-validation-accuracy weights."""
    optim_cfg = cfg["optim"]
    epochs = int(optim_cfg["epochs"])
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    assert train_loader is not None and val_loader is not None

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(optim_cfg.get("label_smoothing", 0.0)))
    optimizer = build_optimizer(model, optim_cfg)
    scheduler = build_scheduler(optimizer, optim_cfg, len(train_loader))

    amp_requested = bool(optim_cfg.get("amp", True))
    # AMP only pays off on CUDA; enabling it on CPU is slower and can be unstable.
    amp_enabled = amp_requested and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    ema_cfg = optim_cfg.get("ema", {}) or {}
    ema = ModelEma(model, float(ema_cfg.get("decay", 0.999))) if ema_cfg.get("enabled") else None

    rng = np.random.default_rng(int(cfg["experiment"]["seed"]))
    tta = str(cfg["eval"].get("tta", "none"))
    patience = int(optim_cfg.get("early_stopping_patience", 0) or 0)

    best_acc = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_used_ema = False
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    start = time.time()

    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            scheduler,
            cfg["augment"],
            float(optim_cfg.get("grad_clip", 0.0) or 0.0),
            rng,
            ema,
        )
        val_stats = evaluate(model, val_loader, device, criterion, tta=tta)
        record = {
            "epoch": epoch,
            **train_stats,
            "val_loss": val_stats["loss"],
            "val_acc": val_stats["acc"],
            "val_top5": val_stats["top5_acc"],
            "elapsed_s": round(time.time() - start, 1),
        }

        # When EMA is on, both candidates are scored and the better one is kept.
        candidate_acc = val_stats["acc"]
        candidate_state = model.state_dict()
        used_ema = False
        if ema is not None:
            ema_stats = evaluate(ema.module, val_loader, device, criterion, tta=tta)
            record["val_acc_ema"] = ema_stats["acc"]
            if ema_stats["acc"] > candidate_acc:
                candidate_acc = ema_stats["acc"]
                candidate_state = ema.module.state_dict()
                used_ema = True

        if candidate_acc > best_acc:
            best_acc = candidate_acc
            best_state = copy.deepcopy(candidate_state)
            best_epoch = epoch
            best_used_ema = used_ema
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        record["best_val_acc"] = best_acc
        history.append(record)
        print(
            f"epoch {epoch:03d}/{epochs} | lr {record['lr']:.2e} | "
            f"train loss {record['train_loss']:.4f} | val loss {record['val_loss']:.4f} | "
            f"val acc {record['val_acc']:.4f} | best {best_acc:.4f} | {record['elapsed_s']:.0f}s",
            flush=True,
        )
        if on_epoch_end is not None:
            on_epoch_end(record)

        if patience and epochs_without_improvement >= patience:
            print(f"Early stopping: no improvement for {patience} epochs.")
            break

    model.load_state_dict(best_state)
    return {
        "model": model,
        "history": history,
        "best_val_acc": best_acc,
        "best_epoch": best_epoch,
        "best_used_ema": best_used_ema,
        "train_time_s": round(time.time() - start, 1),
    }
