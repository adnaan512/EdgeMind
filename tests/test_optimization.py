"""Tests for model compression and optimization."""

import pytest
import torch
import torch.nn as nn

from edgemind.optimization import (
    apply_global_pruning,
    remove_pruning_reparameterization,
    count_zero_weights,
    apply_dynamic_quantization,
    get_model_size_mb
)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3)
        self.fc = nn.Linear(9, 10)
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def test_pruning():
    """Test that magnitude pruning zeroes out weights correctly."""
    model = DummyModel()
    
    # Before pruning
    zeros, total, sparsity = count_zero_weights(model)
    assert sparsity == 0.0
    
    # Apply 50% pruning
    apply_global_pruning(model, amount=0.5)
    
    # Check sparsity (PyTorch pruning uses masks, so weights are technically zeroed during forward)
    # But count_zero_weights looks at the 'weight' attribute, which PyTorch replaces with a property
    # reading from weight_orig and weight_mask.
    zeros, total, sparsity = count_zero_weights(model)
    assert 40.0 <= sparsity <= 60.0  # Should be around 50%
    
    # Remove reparameterization to make zeros permanent
    remove_pruning_reparameterization(model)
    
    # Sparsity should remain
    zeros, total, sparsity_final = count_zero_weights(model)
    assert sparsity_final == sparsity
    
    # The 'weight_mask' buffer should be gone
    assert not hasattr(model.conv, "weight_mask")


def test_quantization_reduces_size():
    """Test that dynamic INT8 quantization reduces model size."""
    # Create a model with a large Linear layer to see the size drop
    model = nn.Sequential(
        nn.Linear(1000, 1000)
    )
    
    original_size = get_model_size_mb(model)
    
    quantized_model = apply_dynamic_quantization(model)
    quantized_size = get_model_size_mb(quantized_model)
    
    # INT8 should be roughly 1/4 the size of FP32
    assert quantized_size < original_size * 0.5


def test_quantization_forward():
    """Test that the quantized model can still process inputs."""
    model = nn.Sequential(
        nn.Linear(10, 5)
    )
    
    quantized_model = apply_dynamic_quantization(model)
    
    x = torch.randn(2, 10)
    # Forward pass should not crash
    out = quantized_model(x)
    assert out.shape == (2, 5)
