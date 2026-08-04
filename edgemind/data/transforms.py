"""
Data transformations and augmentations for EdgeMind AI.

Design Philosophy:
    Augmentation is a crucial part of model training, especially for edge
    devices where we might train smaller models that need heavy regularization.
    
    Instead of hardcoding transforms inside dataset classes, we define a
    factory that builds a `torchvision.transforms.Compose` object from a
    YAML configuration. This makes augmentations fully reproducible and
    experiment-driven.
"""

from __future__ import annotations

from typing import Any, Optional

import torchvision.transforms as T

from edgemind.core.logger import get_logger

logger = get_logger("data.transforms")


def build_transforms(
    config: Optional[dict[str, Any]] = None, 
    is_training: bool = False
) -> T.Compose:
    """Build a torchvision transform pipeline from configuration.

    Args:
        config: A dictionary containing transform settings.
            Expected keys: 'image_size', 'mean', 'std', 'augmentation'.
        is_training: If True, applies data augmentation (RandomCrop, Flip, etc.).
            If False, only applies resizing and normalization.

    Returns:
        A composed torchvision transform.
        
    Example Config:
        data:
          image_size: 224
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]
          augmentation:
            horizontal_flip: true
            random_crop: true
            padding: 4
    """
    config = config or {}
    
    # Default ImageNet statistics
    image_size = config.get("image_size", 224)
    mean = config.get("mean", [0.485, 0.456, 0.406])
    std = config.get("std", [0.229, 0.224, 0.225])
    aug_config = config.get("augmentation", {})

    transforms: list[T.Module] = []

    if is_training:
        # Augmentation pipeline
        if aug_config.get("random_crop", False):
            padding = aug_config.get("padding", 4)
            transforms.append(T.RandomCrop(image_size, padding=padding))
        else:
            transforms.append(T.RandomResizedCrop(image_size))
            
        if aug_config.get("horizontal_flip", True):
            transforms.append(T.RandomHorizontalFlip())
            
        # Add more augmentations here (ColorJitter, AutoAugment) based on config
    else:
        # Validation/Testing pipeline
        # Typically resize slightly larger, then center crop
        resize_dim = int(image_size * 256 / 224)
        transforms.append(T.Resize(resize_dim))
        transforms.append(T.CenterCrop(image_size))

    # ToTensor and Normalize are always applied at the end
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=mean, std=std))

    return T.Compose(transforms)
