"""
Gradient-weighted Class Activation Mapping (Grad-CAM) for EdgeMind AI.

Design Philosophy:
    Edge models deployed in real-world scenarios (medical, industrial) 
    require explainability. Grad-CAM visualizes which pixels the model 
    looked at to make its decision, building trust in black-box models.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from edgemind.core.logger import get_logger

logger = get_logger("explainability.gradcam")


class GradCAM:
    """Computes Grad-CAM for a given PyTorch model and target layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model: The neural network model.
            target_layer: The convolutional layer to inspect.
        """
        self.model = model
        self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
        logger.debug(f"GradCAM initialized on layer: {target_layer.__class__.__name__}")

    def save_activation(self, module, input, output):
        """Forward hook to save the activations."""
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        """Backward hook to save the gradients."""
        self.gradients = grad_output[0]

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """
        Generate the Grad-CAM heatmap.
        
        Args:
            x: Input tensor of shape (1, C, H, W).
            class_idx: The class index to visualize. If None, uses the highest scoring class.
            
        Returns:
            A normalized heatmap as a 2D numpy array [0.0, 1.0].
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        logger.info(f"Generating Grad-CAM for class index: {class_idx}")
        
        # Backward pass for the target class
        self.model.zero_grad()
        target_score = output[0, class_idx]
        target_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]
        
        # Global average pooling on gradients to get channel weights
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted combination of activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # ReLU on CAM (we only care about positive influences)
        cam = np.maximum(cam, 0)
        
        # Normalize
        cam = cam - np.min(cam)
        cam_max = np.max(cam)
        if cam_max != 0:
            cam = cam / cam_max
            
        # Resize to match input image size (H, W)
        _, _, H, W = x.shape
        cam = cv2.resize(cam, (W, H))
        
        return cam

    @staticmethod
    def overlay_heatmap(img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """Overlay the heatmap onto the original image.
        
        Args:
            img: Original image (H, W, 3) in RGB format, normalized [0, 1].
            heatmap: 2D heatmap array [0, 1].
            alpha: Transparency of the heatmap overlay.
            colormap: OpenCV colormap to apply.
            
        Returns:
            The combined image as an RGB numpy array [0, 255] uint8.
        """
        # Convert heatmap to RGB heatmap
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
        # Convert BGR to RGB
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        # Convert original image to uint8 [0, 255]
        img_uint8 = np.uint8(255 * img)
        
        # Blend
        overlay = cv2.addWeighted(img_uint8, 1 - alpha, heatmap_color, alpha, 0)
        return overlay
