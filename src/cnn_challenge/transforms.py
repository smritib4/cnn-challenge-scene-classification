"""Augmentation and preprocessing pipelines.

Three presets let the augmentation strength be varied as a single controlled
factor:

* `none`   - resize only. Reproduces the starter baseline's preprocessing.
* `basic`  - RandomResizedCrop + horizontal flip. The cheap, near-free wins.
* `strong` - adds RandAugment, colour jitter and RandomErasing, aimed at the
             overfitting that 1,920 training images provokes in a large CNN.
"""

from __future__ import annotations

from typing import Any

from torchvision import transforms

PRESETS = {
    "none": {
        "random_resized_crop": False,
        "hflip": 0.0,
        "rand_augment": {"enabled": False},
        "color_jitter": 0.0,
        "random_erasing": 0.0,
    },
    "basic": {
        "random_resized_crop": True,
        "hflip": 0.5,
        "rand_augment": {"enabled": False},
        "color_jitter": 0.0,
        "random_erasing": 0.0,
    },
    "strong": {
        "random_resized_crop": True,
        "hflip": 0.5,
        "rand_augment": {"enabled": True, "num_ops": 2, "magnitude": 9},
        "color_jitter": 0.2,
        "random_erasing": 0.25,
    },
}


def resolve_augment(aug_cfg: dict[str, Any]) -> dict[str, Any]:
    """Expand the named preset, letting explicit keys in the config win."""
    from .config import deep_update

    preset_name = aug_cfg.get("preset", "basic")
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown augmentation preset '{preset_name}'. Choose from {list(PRESETS)}")
    resolved = deep_update(aug_cfg, PRESETS[preset_name])
    # Anything the YAML set explicitly should override the preset, so re-apply
    # the user's own keys on top (excluding the preset marker itself).
    explicit = {k: v for k, v in aug_cfg.items() if k != "preset"}
    return deep_update(resolved, explicit)


def build_transforms(
    data_cfg: dict[str, Any], aug_cfg: dict[str, Any], train: bool
) -> transforms.Compose:
    """Build the train or eval transform pipeline for a given config."""
    img_size = int(data_cfg["img_size"])
    grayscale = bool(data_cfg.get("grayscale", False))
    mean = list(data_cfg["mean"])
    std = list(data_cfg["std"])
    if grayscale and len(mean) == 3:
        # Collapse 3-channel statistics so a colour config can be flipped to
        # grayscale with a single override without crashing in Normalize.
        mean = [sum(mean) / 3.0]
        std = [sum(std) / 3.0]

    ops: list[Any] = []
    if grayscale:
        ops.append(transforms.Grayscale(num_output_channels=1))

    if train:
        aug = resolve_augment(aug_cfg)
        if aug.get("random_resized_crop", True):
            scale = tuple(aug.get("crop_scale", [0.65, 1.0]))
            ops.append(transforms.RandomResizedCrop(img_size, scale=scale, antialias=True))
        else:
            ops.append(transforms.Resize((img_size, img_size), antialias=True))
        if aug.get("hflip", 0.0) > 0:
            ops.append(transforms.RandomHorizontalFlip(p=float(aug["hflip"])))
        ra = aug.get("rand_augment", {}) or {}
        if ra.get("enabled", False):
            # RandAugment operates on PIL/uint8 tensors, so it must precede ToTensor.
            ops.append(
                transforms.RandAugment(
                    num_ops=int(ra.get("num_ops", 2)), magnitude=int(ra.get("magnitude", 9))
                )
            )
        jitter = float(aug.get("color_jitter", 0.0) or 0.0)
        if jitter > 0 and not grayscale:
            ops.append(
                transforms.ColorJitter(
                    brightness=jitter, contrast=jitter, saturation=jitter, hue=jitter / 2
                )
            )
    else:
        # Evaluate on a slightly larger resize then centre crop: the standard
        # ImageNet protocol, and it matches the object scale the crops produce.
        resize = int(round(img_size * 1.14))
        ops.append(transforms.Resize(resize, antialias=True))
        ops.append(transforms.CenterCrop(img_size))

    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(mean=mean, std=std))

    if train:
        erasing = float(resolve_augment(aug_cfg).get("random_erasing", 0.0) or 0.0)
        if erasing > 0:
            # RandomErasing works on normalised tensors, hence its position last.
            ops.append(transforms.RandomErasing(p=erasing, value="random"))

    return transforms.Compose(ops)
