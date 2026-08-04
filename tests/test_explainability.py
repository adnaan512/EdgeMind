"""Tests for Grad-CAM explainability module."""

import numpy as np
import torch
import torch.nn as nn

from edgemind.explainability import GradCAM


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def test_gradcam_heatmap_generation():
    """Test that GradCAM generates a valid heatmap [0, 1]."""
    model = DummyModel()
    
    # Target the conv1 layer
    cam = GradCAM(model, model.conv1)
    
    # Dummy image input (Batch=1, Channels=3, H=32, W=32)
    x = torch.randn(1, 3, 32, 32)
    
    # Generate heatmap
    heatmap = cam(x)
    
    # Verify shape matches input spatial dimensions
    assert heatmap.shape == (32, 32)
    
    # Verify values are normalized between 0 and 1
    assert np.min(heatmap) >= 0.0
    assert np.max(heatmap) <= 1.0


def test_gradcam_overlay():
    """Test that the heatmap can be overlaid on an RGB image."""
    # Dummy RGB image [0, 1]
    img = np.random.rand(32, 32, 3).astype(np.float32)
    
    # Dummy heatmap [0, 1]
    heatmap = np.random.rand(32, 32).astype(np.float32)
    
    overlay = GradCAM.overlay_heatmap(img, heatmap, alpha=0.5)
    
    # Verify overlay is an RGB uint8 image [0, 255]
    assert overlay.shape == (32, 32, 3)
    assert overlay.dtype == np.uint8
    assert np.min(overlay) >= 0
    assert np.max(overlay) <= 255
