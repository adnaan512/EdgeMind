"""Tests for the data transform factory."""

import torchvision.transforms as T

from edgemind.data.transforms import build_transforms


def test_build_transforms_default():
    """Should build a default validation transform with ImageNet stats."""
    transform = build_transforms(is_training=False)
    
    assert isinstance(transform, T.Compose)
    # Resize, CenterCrop, ToTensor, Normalize
    assert len(transform.transforms) == 4
    
    assert isinstance(transform.transforms[0], T.Resize)
    assert isinstance(transform.transforms[3], T.Normalize)
    
    # ImageNet defaults
    assert transform.transforms[3].mean == [0.485, 0.456, 0.406]


def test_build_transforms_training():
    """Should include augmentations when is_training is True."""
    config = {
        "image_size": 32,
        "augmentation": {
            "horizontal_flip": True,
            "random_crop": True,
            "padding": 4
        }
    }
    transform = build_transforms(config, is_training=True)
    
    assert isinstance(transform, T.Compose)
    
    # RandomCrop, RandomHorizontalFlip, ToTensor, Normalize
    assert len(transform.transforms) == 4
    
    assert isinstance(transform.transforms[0], T.RandomCrop)
    assert transform.transforms[0].padding == 4
    assert isinstance(transform.transforms[1], T.RandomHorizontalFlip)


def test_build_transforms_custom_stats():
    """Should use provided mean and std."""
    config = {
        "mean": [0.5, 0.5, 0.5],
        "std": [0.1, 0.1, 0.1]
    }
    transform = build_transforms(config, is_training=False)
    
    normalize = transform.transforms[-1]
    assert normalize.mean == [0.5, 0.5, 0.5]
    assert normalize.std == [0.1, 0.1, 0.1]
