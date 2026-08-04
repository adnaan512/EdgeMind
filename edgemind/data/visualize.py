"""
Data visualization utilities.

Design Philosophy:
    Before training any model, a researcher must LOOK at the data.
    Incorrect augmentations or normalizations ruin experiments silently.
    This module provides tools to visualize a batch of augmented data.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from edgemind.core.logger import get_logger

logger = get_logger("data.visualize")


def unnormalize(
    tensor: torch.Tensor,
    mean: list[float] = [0.485, 0.456, 0.406],
    std: list[float] = [0.229, 0.224, 0.225]
) -> torch.Tensor:
    """Reverse the normalization applied during transforms.
    
    Useful for displaying images that have been normalized.
    """
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    return tensor * std_t + mean_t


def show_batch(
    dataloader: DataLoader,
    num_images: int = 16,
    mean: list[float] = [0.485, 0.456, 0.406],
    std: list[float] = [0.229, 0.224, 0.225],
    save_path: str | None = None
) -> None:
    """Visualize a batch of images from a dataloader.
    
    Args:
        dataloader: The PyTorch dataloader.
        num_images: Number of images to show.
        mean: Normalization mean to reverse.
        std: Normalization std to reverse.
        save_path: If provided, saves the plot to this file instead of displaying.
    """
    # Get a batch
    images, labels = next(iter(dataloader))
    
    # Truncate if batch is larger than requested
    images = images[:num_images]
    labels = labels[:num_images]
    
    # Unnormalize
    images = unnormalize(images, mean, std)
    
    # Clip to [0, 1] just in case of rounding errors
    images = torch.clamp(images, 0, 1)
    
    # Make a grid
    grid = torchvision.utils.make_grid(images, nrow=int(np.sqrt(num_images)))
    
    # Convert to numpy and HWC format for matplotlib
    grid_np = grid.numpy().transpose((1, 2, 0))
    
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_np)
    plt.axis("off")
    
    dataset = dataloader.dataset
    if hasattr(dataset, "classes"):
        classes = dataset.classes
        title = " | ".join([classes[l.item()] for l in labels])
        # Only show title if it's not too long
        if len(labels) <= 8:
            plt.title(title)
            
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved batch visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()
