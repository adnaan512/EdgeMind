"""
Data pipeline components for EdgeMind AI.

This module exports datasets and the dataloader factory,
and registers datasets into the DATASETS registry.
"""

from edgemind.core.registry import Registry
from edgemind.data.base_dataset import BaseDataset
from edgemind.data.cifar_dataset import CIFAR10Dataset
from edgemind.data.folder_dataset import CustomFolderDataset
from edgemind.data.transforms import build_transforms

# Create the registry for datasets
DATASETS = Registry("datasets")

# Register datasets
DATASETS.register("cifar10")(CIFAR10Dataset)
DATASETS.register("custom_folder")(CustomFolderDataset)

__all__ = [
    "DATASETS",
    "BaseDataset",
    "CIFAR10Dataset",
    "CustomFolderDataset",
    "build_transforms",
]
