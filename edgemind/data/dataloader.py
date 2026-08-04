"""
DataLoader factory for EdgeMind AI.

Design Philosophy:
    Creating dataloaders involves wiring together datasets, transforms,
    and multiprocessing settings. This factory abstracts that away, allowing
    a single function call to return fully configured train/val dataloaders
    based entirely on the YAML config.
"""

from __future__ import annotations

from typing import Any, Optional

from torch.utils.data import DataLoader

from edgemind.core.logger import get_logger
from edgemind.data import DATASETS
from edgemind.data.transforms import build_transforms

logger = get_logger("data.loader")


def build_dataloaders(
    config: Any,
) -> tuple[DataLoader, DataLoader]:
    """Build training and validation dataloaders from configuration.

    Args:
        config: EdgeMindConfig object containing `data` and `training` sections.

    Returns:
        A tuple of (train_loader, val_loader).
    """
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    
    dataset_name = data_cfg.get("dataset", "cifar10")
    batch_size = train_cfg.get("batch_size", 32)
    num_workers = data_cfg.get("num_workers", 2)
    pin_memory = data_cfg.get("pin_memory", True)
    
    logger.info(f"Building dataloaders for dataset: {dataset_name}")
    
    # Build transforms
    train_transform = build_transforms(data_cfg, is_training=True)
    val_transform = build_transforms(data_cfg, is_training=False)
    
    # Prepare dataset kwargs from config
    # We pass the relevant config dict to the registry build method
    # Since our datasets expect `split` and `transform`, we add them
    
    # For train
    train_kwargs = data_cfg.to_dict() if hasattr(data_cfg, "to_dict") else data_cfg.copy()
    train_kwargs["name"] = dataset_name
    train_kwargs["split"] = "train"
    train_kwargs["transform"] = train_transform
    # Clean up non-dataset kwargs if they exist (e.g., num_workers, augmentation)
    train_kwargs.pop("num_workers", None)
    train_kwargs.pop("pin_memory", None)
    train_kwargs.pop("augmentation", None)
    train_kwargs.pop("image_size", None)
    train_kwargs.pop("mean", None)
    train_kwargs.pop("std", None)
    train_kwargs.pop("dataset", None)

    # For val
    val_kwargs = train_kwargs.copy()
    val_kwargs["split"] = "val"
    val_kwargs["transform"] = val_transform

    # Instantiate datasets via registry
    train_dataset = DATASETS.build(train_kwargs)
    val_dataset = DATASETS.build(val_kwargs)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    
    logger.info(
        f"Dataloaders ready. Train: {len(train_loader)} batches. "
        f"Val: {len(val_loader)} batches."
    )
    
    return train_loader, val_loader
