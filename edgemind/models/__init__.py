"""
Model zoo components for EdgeMind AI.

This module exports the base model, backbones, and the main image classifier,
and registers the models into the MODELS registry.
"""

from edgemind.core.registry import Registry
from edgemind.models.base_model import BaseEdgeModel
from edgemind.models.classifier import ImageClassifier

# Create the registry for models
MODELS = Registry("models")

# Register the standard image classifier
MODELS.register("image_classifier")(ImageClassifier)

__all__ = [
    "MODELS",
    "BaseEdgeModel",
    "ImageClassifier",
]
