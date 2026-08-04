"""
Custom image folder dataset for EdgeMind AI.

Design Philosophy:
    While CIFAR-10 is great for benchmarking, real-world edge AI (manufacturing,
    medical, retail) requires training on custom data.
    
    This dataset wraps PyTorch's ImageFolder, allowing users to train on any
    directory structured as:
    root/
      train/
        class1/
        class2/
      val/
        class1/
        class2/
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from torchvision.datasets import ImageFolder

from edgemind.core.logger import get_logger
from edgemind.data.base_dataset import BaseDataset

logger = get_logger("data.folder")


class CustomFolderDataset(BaseDataset):
    """Dataset for custom image folders."""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Any = None,
    ):
        """Initialize the folder dataset.
        
        Args:
            data_dir: Root directory containing 'train' and 'val' subfolders.
            split: "train", "val", or "test".
            transform: Transform pipeline.
        """
        super().__init__(split=split, transform=transform)
        
        split_dir = Path(data_dir) / split
        
        if not split_dir.exists() or not split_dir.is_dir():
            raise FileNotFoundError(
                f"Dataset directory '{split_dir}' not found. "
                f"Expected structure: {data_dir}/{{train,val,test}}/<classes>/"
            )
            
        self.dataset = ImageFolder(
            root=str(split_dir),
            transform=self.transform
        )
        
        logger.info(
            f"Loaded {split} dataset from {split_dir}: "
            f"{len(self.dataset)} images, {len(self.classes)} classes."
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        return self.dataset[idx]
        
    @property
    def classes(self) -> list[str]:
        return self.dataset.classes
