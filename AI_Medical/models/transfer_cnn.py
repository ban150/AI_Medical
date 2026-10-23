"""
CNN binaire avec backbone ImageNet gele (transfer learning) en PyTorch.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torchvision import models


def _freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def build_binary_transfer_model(
    backbone_name: str,
    learning_rate: float = 1e-3,
    dropout: float = 0.5,
) -> tuple[nn.Module, torch.optim.Optimizer, nn.Module]:
    """
    Cree un modele binaire (logits) + optimizer + BCEWithLogitsLoss.

    Sortie modele: tenseur shape (B,) de logits pour la classe PNEUMONIA (label=1).
    """
    name = backbone_name.lower()
    if name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        _freeze(model)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, 1))
    elif name == "mobilenetv2":
        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)
        _freeze(model.features)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, 1))
    else:
        raise ValueError(f"Backbone non supporte: {backbone_name}")

    criterion = nn.BCEWithLogitsLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
    return model, optimizer, criterion


def get_backbone_factories() -> dict[str, Callable[[], str]]:
    """Registre des backbones utilises dans le memoire."""
    return {
        "ResNet50": lambda: "resnet50",
        "MobileNetV2": lambda: "mobilenetv2",
    }
