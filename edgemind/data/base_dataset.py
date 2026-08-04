"""
Base dataset class for EdgeMind AI.

Design Philosophy:
    All datasets in our framework must adhere to a common interface. This
    ensures that our dataloader factory can instantiate them uniformly.
    
    By inheriting from `torch.utils.data.Dataset`, our custom datasets
    remain fully compatible with standard PyTorch infrastructure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """Abstract base class for all EdgeMind AI datasets.
    
    Subclasses must implement `__len__` and `__getitem__`.
    """

    def __init__(self, split: str = "train", transform: Any = None):
        """Initialize the base dataset.
        
        Args:
            split: The dataset split ("train", "val", or "test").
            transform: The transform pipeline to apply to the images.
        """
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', or 'test'.")
            
        self.split = split
        self.transform = transform

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        """Fetch the sample and target at the given index.
        
        Returns:
            A tuple (image, label). The image should have `self.transform` applied.
        """
        pass
        
    @property
    @abstractmethod
    def classes(self) -> list[str]:
        """Return a list of class names."""
        pass
