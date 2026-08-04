"""Tests for the model zoo and transfer learning wrappers."""

import torch

from edgemind.models import MODELS
from edgemind.models.classifier import ImageClassifier


def test_registry_contains_models():
    """Ensure models are registered properly."""
    assert "image_classifier" in MODELS


def test_image_classifier_mobilenet():
    """Test building MobileNetV2 classifier."""
    model = MODELS.build({
        "name": "image_classifier",
        "backbone_name": "mobilenet_v2",
        "num_classes": 10,
        "pretrained": False, # False for faster tests
        "freeze_features": False,
    })
    
    assert isinstance(model, ImageClassifier)
    
    # Check forward pass
    x = torch.randn(2, 3, 32, 32) # batch_size=2, 3 channels, 32x32 image (CIFAR size)
    out = model(x)
    
    assert out.shape == (2, 10)


def test_image_classifier_resnet():
    """Test building ResNet18 classifier."""
    model = ImageClassifier(
        backbone_name="resnet18",
        num_classes=5,
        pretrained=False
    )
    
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    
    assert out.shape == (2, 5)


def test_freeze_unfreeze():
    """Test freezing and unfreezing the backbone."""
    model = ImageClassifier(backbone_name="mobilenet_v2", pretrained=False)
    
    # Initially unfrozen
    trainable_before = model.count_parameters(trainable_only=True)
    
    # Freeze backbone
    model.freeze_backbone()
    trainable_after_freeze = model.count_parameters(trainable_only=True)
    
    # Only head parameters should be trainable
    assert trainable_after_freeze < trainable_before
    
    # Head should still be trainable
    for param in model.head.parameters():
        assert param.requires_grad is True
        
    # Backbone should not be trainable
    for param in model.backbone.parameters():
        assert param.requires_grad is False
        
    # Unfreeze
    model.unfreeze_backbone()
    trainable_after_unfreeze = model.count_parameters(trainable_only=True)
    assert trainable_after_unfreeze == trainable_before
