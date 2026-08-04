"""
Pretrained backbone models for transfer learning.

Design Philosophy:
    Edge AI relies heavily on lightweight architectures like MobileNet and
    EfficientNet. Instead of rewriting them, we wrap PyTorch's official
    implementations.
    
    We strip the final classification layer so these models act purely as
    feature extractors. This makes it trivial to attach custom heads for
    new datasets.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import (
    mobilenet_v2, MobileNet_V2_Weights,
    resnet18, ResNet18_Weights
)

from edgemind.core.logger import get_logger

logger = get_logger("models.backbones")


class BackboneWrapper(nn.Module):
    """Wraps a torchvision model to act purely as a feature extractor."""
    
    def __init__(self, model: nn.Module, feature_dim: int):
        super().__init__()
        self.features = model
        self.feature_dim = feature_dim
        
    def forward(self, x):
        return self.features(x)


def get_mobilenet_v2(pretrained: bool = True) -> BackboneWrapper:
    """Load MobileNetV2 backbone (excellent for edge devices)."""
    weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = mobilenet_v2(weights=weights)
    
    # Feature dimension of MobileNetV2 is 1280
    feature_dim = model.last_channel
    
    # Replace the classifier with an Identity layer
    model.classifier = nn.Identity()
    
    if pretrained:
        logger.info("Loaded pretrained MobileNetV2 backbone")
        
    return BackboneWrapper(model, feature_dim)


def get_resnet18(pretrained: bool = True) -> BackboneWrapper:
    """Load ResNet18 backbone (standard baseline)."""
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    
    # Feature dimension of ResNet18 is 512
    feature_dim = model.fc.in_features
    
    # Replace the fully connected layer with Identity
    model.fc = nn.Identity()
    
    if pretrained:
        logger.info("Loaded pretrained ResNet18 backbone")
        
    return BackboneWrapper(model, feature_dim)
