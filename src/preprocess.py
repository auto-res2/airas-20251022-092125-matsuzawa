"""Data loading & preprocessing utilities (TorchVision CIFAR-10).

All datasets and downloaded files are stored strictly under ./.cache/ as per
specification.  This module exposes `get_dataloaders(cfg)` which returns
(train_loader, val_loader, test_loader) according to the split ratios in
`cfg.dataset.train_val_split`.
"""

import os
from pathlib import Path
from typing import Tuple

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

CACHE_ROOT = Path(".cache").absolute()
DATASETS_ROOT = CACHE_ROOT / "datasets"
DATASETS_ROOT.mkdir(parents=True, exist_ok=True)

os.environ["TORCH_HOME"] = str(CACHE_ROOT / "torch")

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _build_transforms(aug_cfg: DictConfig, is_train: bool):
    t_list = []
    if is_train and aug_cfg is not None:
        if "random_crop" in aug_cfg:
            rc = aug_cfg.random_crop
            t_list.append(transforms.RandomCrop(rc.size, padding=rc.padding))
        if "random_horizontal_flip" in aug_cfg:
            prob = float(aug_cfg.random_horizontal_flip)
            t_list.append(transforms.RandomHorizontalFlip(prob))
    t_list.append(transforms.ToTensor())
    if aug_cfg is not None and "normalization" in aug_cfg:
        mean, std = aug_cfg.normalization.mean, aug_cfg.normalization.std
        t_list.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(t_list)

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def get_dataloaders(cfg: DictConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return train/val/test DataLoader triplet according to the config."""
    dataset_name = str(cfg.dataset.name).lower()
    if dataset_name not in {"cifar-10", "cifar10"}:
        raise ValueError(f"Unsupported dataset: {cfg.dataset.name}")

    # Ensure we always download/copy to .cache
    data_root = DATASETS_ROOT / dataset_name
    data_root.mkdir(parents=True, exist_ok=True)

    train_tf = _build_transforms(cfg.dataset.get("augmentation", None), True)
    test_tf = _build_transforms(cfg.dataset.get("augmentation", None), False)

    if dataset_name in {"cifar-10", "cifar10"}:
        train_full = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_tf)
        test_set = datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_tf)

    train_ratio, val_ratio = cfg.dataset.train_val_split
    total_len = len(train_full)
    len_train = int(total_len * train_ratio)
    len_val = total_len - len_train

    generator = torch.Generator().manual_seed(int(cfg.get("seed", 42)))
    train_set, val_set = random_split(train_full, [len_train, len_val], generator=generator)

    dl_kwargs = dict(
        batch_size=int(cfg.training.batch_size),
        num_workers=int(cfg.training.num_workers),
        pin_memory=True,
    )
    train_loader = DataLoader(train_set, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **dl_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **dl_kwargs)
    return train_loader, val_loader, test_loader
