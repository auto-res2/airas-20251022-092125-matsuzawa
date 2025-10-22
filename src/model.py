"""Model factory – currently supports ResNet-18 (TorchVision).

Call build_model(cfg) with the *run* sub-config (has .model & .dataset).
"""

from omegaconf import DictConfig
import torch.nn as nn
from torchvision import models


def build_model(cfg: DictConfig) -> nn.Module:
    name = str(cfg.model.name).lower()
    num_classes = int(cfg.dataset.num_classes)
    pretrained = bool(cfg.model.get("pretrained", False))

    if name in {"resnet-18", "resnet18"}:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {cfg.model.name}")
    return model
