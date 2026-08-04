"""
CIFAR-10 Dataset wrapper for EdgeMind AI.

Design Philosophy:
    CIFAR-10 is the canonical dataset for edge AI and model compression
    research. By wrapping the standard torchvision dataset, we integrate
    it seamlessly into our config-driven pipeline.
"""

from __future__ import annotations

import os
from typing import Any

from torchvision.datasets import CIFAR10

from edgemind.core.logger import get_logger
from edgemind.data.base_dataset import BaseDataset

logger = get_logger("data.cifar")


class CIFAR10Dataset(BaseDataset):
    """CIFAR-10 dataset implementation."""

    def __init__(
        self,
        data_dir: str = "./data",
        split: str = "train",
        transform: Any = None,
        download: bool = True,
    ):
        """Initialize the CIFAR-10 dataset.
        
        Args:
            data_dir: Root directory for downloading/storing the data.
            split: "train" or "val" (CIFAR-10 standard split).
            transform: Transform pipeline.
            download: Whether to download if not present.
        """
        super().__init__(split=split, transform=transform)
        
        is_train = split == "train"
        
        # Ensure directory exists to avoid download errors in some environments
        os.makedirs(data_dir, exist_ok=True)
        
        try:
            self.dataset = CIFAR10(
                root=data_dir,
                train=is_train,
                transform=self.transform,
                download=download,
            )
        except Exception as e:
            logger.error(f"Failed to load CIFAR-10 dataset: {e}")
            raise

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        return self.dataset[idx]
        
    @property
    def classes(self) -> list[str]:
        return [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]
