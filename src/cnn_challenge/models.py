"""Model definitions and the backbone factory.

Everything here is a convolutional network, as the assignment requires. Two
from-scratch models are kept for reference points, and the rest are ImageNet
pretrained torchvision CNNs whose classifier head is replaced with a fresh
16-way linear layer.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torchvision import models


class TNet(nn.Module):
    """The starter baseline, reproduced verbatim so its number is comparable.

    One 3x3 conv with 16 filters, a 4x4 max-pool, then a linear layer. There is
    no normalisation, no depth and no pooling hierarchy, so it cannot build the
    mid-level features scene recognition needs.
    """

    def __init__(self, num_classes: int = 16, in_channels: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4),
        )
        # Lazily sized so the model works at any input resolution.
        self.classifier = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class SmallCNN(nn.Module):
    """A modern from-scratch CNN: the honest 'no pretraining' comparison.

    Four stages of (conv-BN-ReLU) x2 + max-pool with global average pooling.
    Batch norm and GAP are the two pieces the starter lacks that matter most,
    and it exists to isolate how much of the final gain comes from ImageNet
    initialisation rather than simply from a better architecture.
    """

    def __init__(self, num_classes: int = 16, in_channels: int = 3, dropout: float = 0.2) -> None:
        super().__init__()
        widths = (64, 128, 256, 512)
        blocks: list[nn.Module] = []
        prev = in_channels
        for width in widths:
            blocks.extend(
                [
                    nn.Conv2d(prev, width, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(width),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(width),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                ]
            )
            prev = width
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(widths[-1], num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


# Backbone name -> (torchvision constructor, default weights enum attribute)
TORCHVISION_BACKBONES: dict[str, tuple[Any, str]] = {
    "resnet18": (models.resnet18, "ResNet18_Weights"),
    "resnet34": (models.resnet34, "ResNet34_Weights"),
    "resnet50": (models.resnet50, "ResNet50_Weights"),
    "resnext50_32x4d": (models.resnext50_32x4d, "ResNeXt50_32X4D_Weights"),
    "wide_resnet50_2": (models.wide_resnet50_2, "Wide_ResNet50_2_Weights"),
    "densenet121": (models.densenet121, "DenseNet121_Weights"),
    "efficientnet_b0": (models.efficientnet_b0, "EfficientNet_B0_Weights"),
    "efficientnet_v2_s": (models.efficientnet_v2_s, "EfficientNet_V2_S_Weights"),
    "convnext_tiny": (models.convnext_tiny, "ConvNeXt_Tiny_Weights"),
    "mobilenet_v3_large": (models.mobilenet_v3_large, "MobileNet_V3_Large_Weights"),
}


def _replace_head(model: nn.Module, num_classes: int, dropout: float) -> nn.Module:
    """Swap the ImageNet 1000-way head for a fresh `num_classes` linear layer.

    torchvision families expose the head differently (`fc` vs `classifier` vs a
    `Sequential` ending in a `Linear`), so each case is handled explicitly
    instead of guessing.
    """
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_features = model.fc.in_features
        model.fc = _head(in_features, num_classes, dropout)
        return model

    if hasattr(model, "classifier"):
        classifier = model.classifier
        if isinstance(classifier, nn.Linear):  # densenet
            model.classifier = _head(classifier.in_features, num_classes, dropout)
            return model
        if isinstance(classifier, nn.Sequential):  # efficientnet, convnext, mobilenet
            for idx in range(len(classifier) - 1, -1, -1):
                if isinstance(classifier[idx], nn.Linear):
                    in_features = classifier[idx].in_features
                    classifier[idx] = nn.Linear(in_features, num_classes)
                    if dropout > 0:
                        # Reuse the family's own dropout slot when it has one.
                        for module in classifier:
                            if isinstance(module, nn.Dropout):
                                module.p = dropout
                    return model
    raise ValueError(f"Could not locate a classifier head on {type(model).__name__}")


def _head(in_features: int, num_classes: int, dropout: float) -> nn.Module:
    if dropout > 0:
        return nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    return nn.Linear(in_features, num_classes)


def _adapt_first_conv(model: nn.Module) -> None:
    """Convert the stem to single-channel input by summing the RGB filters.

    Summing (rather than re-initialising) preserves the pretrained filter
    responses for a grayscale image, which is the standard trick when feeding
    1-channel data to an ImageNet backbone.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            new_conv = nn.Conv2d(
                1,
                module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                bias=module.bias is not None,
            )
            with torch.no_grad():
                new_conv.weight.copy_(module.weight.sum(dim=1, keepdim=True))
                if module.bias is not None:
                    new_conv.bias.copy_(module.bias)
            parent = model
            parts = name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_conv)
            return


def classifier_parameter_names(model: nn.Module) -> set[str]:
    """Names of the freshly initialised head parameters.

    Used to give the head a larger learning rate than the pretrained backbone.
    """
    names: set[str] = set()
    for attr in ("fc", "classifier", "head"):
        module = getattr(model, attr, None)
        if isinstance(module, nn.Module):
            names.update(f"{attr}.{n}" for n, _ in module.named_parameters())
    return names


def build_model(cfg: dict[str, Any], num_classes: int) -> nn.Module:
    """Instantiate the model described by `cfg['model']`."""
    model_cfg = cfg["model"]
    name = str(model_cfg["name"]).lower()
    pretrained = bool(model_cfg.get("pretrained", True))
    dropout = float(model_cfg.get("dropout", 0.0) or 0.0)
    in_channels = 1 if cfg["data"].get("grayscale", False) else 3

    if name == "tnet":
        return TNet(num_classes=num_classes, in_channels=in_channels)
    if name in {"smallcnn", "small_cnn"}:
        return SmallCNN(num_classes=num_classes, in_channels=in_channels, dropout=dropout)

    if name not in TORCHVISION_BACKBONES:
        raise ValueError(
            f"Unknown model '{name}'. Available: "
            f"{['tnet', 'smallcnn'] + sorted(TORCHVISION_BACKBONES)}"
        )

    constructor, weights_attr = TORCHVISION_BACKBONES[name]
    weights = None
    if pretrained:
        weights = getattr(models, weights_attr).DEFAULT
    model = constructor(weights=weights)
    model = _replace_head(model, num_classes, dropout)

    if in_channels == 1:
        _adapt_first_conv(model)

    freeze = str(model_cfg.get("freeze", "none")).lower()
    if freeze == "backbone":
        head_names = classifier_parameter_names(model)
        for param_name, param in model.named_parameters():
            param.requires_grad = param_name in head_names
    elif freeze not in {"none", ""}:
        raise ValueError(f"Unsupported freeze mode '{freeze}' (use 'none' or 'backbone')")

    return model
