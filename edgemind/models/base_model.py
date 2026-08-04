"""
Base model interfaces for EdgeMind AI.

Design Philosophy:
    While we use standard PyTorch `nn.Module` objects, wrapping them in a
    base class allows us to enforce certain behaviors useful for research:
    - Freezing/unfreezing backbones for transfer learning
    - Exposing the feature dimension size automatically
    - Standardizing weight initialization
"""

from __future__ import annotations

import torch
import torch.nn as nn

from edgemind.core.logger import get_logger

logger = get_logger("models.base")


class BaseEdgeModel(nn.Module):
    """Abstract base class for all EdgeMind AI models."""

    def __init__(self):
        super().__init__()

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count the number of parameters in the model.
        
        Args:
            trainable_only: If True, only counts parameters that require gradients.
            
        Returns:
            Total number of parameters.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def freeze_backbone(self) -> None:
        """Freeze the backbone network (prevent gradient updates).
        
        Must be implemented by subclasses that have a backbone.
        """
        raise NotImplementedError("Subclasses must implement freeze_backbone()")

    def unfreeze_backbone(self) -> None:
        """Unfreeze the backbone network."""
        raise NotImplementedError("Subclasses must implement unfreeze_backbone()")
