"""Build every registered model and verify it before any training happens.

    python scripts/check_models.py            # fast: random init only
    python scripts/check_models.py --pretrained   # also downloads ImageNet weights

This exists because of a real bug. `TNet` was originally written with
`nn.LazyLinear` so it would adapt to any input resolution, but lazy parameters do
not exist until the first forward pass, so `train.py` crashed when it counted
parameters beforehand. The smoke config used `resnet18`, so the `tnet` path was
never exercised and the failure only appeared on the GPU run.

The check therefore deliberately does things **in the order train.py does them**:
count parameters first, build an optimizer second, and only then run a forward
pass. Checking shapes after a forward pass would not have caught the bug.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch  # noqa: E402

from cnn_challenge.config import DEFAULTS, deep_update  # noqa: E402
from cnn_challenge.engine import build_optimizer  # noqa: E402
from cnn_challenge.models import TORCHVISION_BACKBONES, build_model  # noqa: E402
from cnn_challenge.utils import count_parameters  # noqa: E402

NUM_CLASSES = 16


def check_one(name: str, img_size: int, grayscale: bool, pretrained: bool, freeze: str) -> str:
    cfg = deep_update(
        DEFAULTS,
        {
            "data": {"img_size": img_size, "grayscale": grayscale},
            "model": {"name": name, "pretrained": pretrained, "freeze": freeze},
        },
    )
    model = build_model(cfg, num_classes=NUM_CLASSES)

    # Order matters: this is what train.py does, and what previously failed.
    n_params = count_parameters(model)
    build_optimizer(model, cfg["optim"])

    channels = 1 if grayscale else 3
    with torch.no_grad():
        logits = model(torch.randn(2, channels, img_size, img_size))
    if tuple(logits.shape) != (2, NUM_CLASSES):
        raise AssertionError(f"expected logits (2, {NUM_CLASSES}), got {tuple(logits.shape)}")
    return f"params={n_params:>11,}  logits={tuple(logits.shape)}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Also load ImageNet weights (slower; downloads on first use)",
    )
    args = parser.parse_args()

    cases: list[tuple[str, int, bool, str]] = [
        # The starter at its own resolution, and at the resolution experiment 01 uses.
        ("tnet", 64, True, "none"),
        ("tnet", 128, False, "none"),
        ("smallcnn", 128, False, "none"),
        ("smallcnn", 64, True, "none"),
    ]
    for backbone in sorted(TORCHVISION_BACKBONES):
        cases.append((backbone, 224, False, "none"))
    # Grayscale stem adaptation and the frozen-backbone path.
    cases.append(("resnet18", 224, True, "none"))
    cases.append(("resnet18", 224, False, "backbone"))

    failures = 0
    for name, img_size, grayscale, freeze in cases:
        label = f"{name:<20} {img_size:>4}px {'gray' if grayscale else 'rgb '} freeze={freeze:<8}"
        try:
            detail = check_one(name, img_size, grayscale, args.pretrained, freeze)
            print(f"PASS  {label} {detail}")
        except Exception as error:  # noqa: BLE001 - report all, keep checking
            failures += 1
            print(f"FAIL  {label} {type(error).__name__}: {error}")
            traceback.print_exc()

    print(f"\n{len(cases) - failures}/{len(cases)} passed")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
