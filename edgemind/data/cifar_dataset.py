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
        kaggle_dataset: str | None = None,
    ):
        """Initialize the CIFAR-10 dataset.
        
        Args:
            data_dir: Root directory for downloading/storing the data.
            split: "train" or "val" (CIFAR-10 standard split).
            transform: Transform pipeline.
            download: Whether to download if not present.
            kaggle_dataset: Optional Kaggle dataset string (e.g., "adnanhassnain/cifar-10-python") for faster downloads.
        """
        super().__init__(split=split, transform=transform)
        
        is_train = split == "train"
        
        # Ensure directory exists to avoid download errors in some environments
        os.makedirs(data_dir, exist_ok=True)
        
        if download and kaggle_dataset:
            try:
                import kagglehub
                import shutil
                from pathlib import Path
                
                logger.info(f"Downloading CIFAR-10 from Kaggle: {kaggle_dataset}")
                kaggle_path = kagglehub.dataset_download(kaggle_dataset)
                
                target_dir = Path(data_dir) / "cifar-10-batches-py"
                if not target_dir.exists():
                    target_dir.mkdir(parents=True, exist_ok=True)
                    src_dir = Path(kaggle_path)
                    
                    # Handle case where files are inside a subfolder
                    if (src_dir / "cifar-10-batches-py").exists():
                        src_dir = src_dir / "cifar-10-batches-py"
                        
                    for item in src_dir.iterdir():
                        if item.is_file():
                            shutil.copy2(item, target_dir)
                    logger.info(f"Copied Kaggle dataset to {target_dir}")
                
                # Disable standard torchvision download since it's already local
                download = False
            except ImportError:
                logger.warning("kagglehub not installed. Falling back to torchvision download. Run `pip install kagglehub`.")
            except Exception as e:
                logger.warning(f"Kaggle download failed: {e}. Falling back to torchvision download.")

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
