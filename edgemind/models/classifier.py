"""
Image Classifier wrapper for transfer learning.

Design Philosophy:
    This is the core model component. It combines a feature-extracting
    backbone with a custom classification head. 
    
    This design is critical for research: it allows us to swap backbones
    (MobileNet vs ResNet) via config without changing the training logic,
    and makes it trivial to freeze the backbone for transfer learning.
"""

from __future__ import annotations

import torch.nn as nn

from edgemind.core.logger import get_logger
from edgemind.models.base_model import BaseEdgeModel
from edgemind.models.backbones import get_mobilenet_v2, get_resnet18

logger = get_logger("models.classifier")


class ImageClassifier(BaseEdgeModel):
    """Standard image classification model with configurable backbone."""

    def __init__(
        self,
        backbone_name: str = "mobilenet_v2",
        num_classes: int = 10,
        pretrained: bool = True,
        freeze_features: bool = False,
        dropout_rate: float = 0.2,
    ):
        """Initialize the classifier.
        
        Args:
            backbone_name: Name of the backbone architecture.
            num_classes: Number of output classes for the dataset.
            pretrained: Whether to load ImageNet weights.
            freeze_features: If True, freezes the backbone for fine-tuning.
            dropout_rate: Dropout probability before the final linear layer.
        """
        super().__init__()
        
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        
        # 1. Load the backbone feature extractor
        if backbone_name == "mobilenet_v2":
            self.backbone = get_mobilenet_v2(pretrained=pretrained)
        elif backbone_name == "resnet18":
            self.backbone = get_resnet18(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
            
        feature_dim = self.backbone.feature_dim
        
        # 2. Build the classification head
        self.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(feature_dim, num_classes)
        )
        
        # 3. Apply transfer learning settings
        if freeze_features:
            self.freeze_backbone()
            
        logger.info(
            f"Built ImageClassifier(backbone={backbone_name}, "
            f"classes={num_classes}, pretrained={pretrained}). "
            f"Total params: {self.count_parameters():,}"
        )

    def forward(self, x):
        """Forward pass."""
        features = self.backbone(x)
        return self.head(features)

    def freeze_backbone(self) -> None:
        """Freeze the backbone network (prevent gradient updates)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen for transfer learning.")

    def unfreeze_backbone(self) -> None:
        """Unfreeze the backbone network."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen for fine-tuning.")
