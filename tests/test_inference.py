"""Tests for ONNX export and inference profiling."""

import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from edgemind.inference import export_to_onnx, profile_latency


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 10)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def test_export_to_onnx(tmp_path):
    """Test that PyTorch models can be successfully exported to ONNX format."""
    model = DummyModel()
    
    out_file = tmp_path / "test_model.onnx"
    
    # Export
    result_path = export_to_onnx(
        model, 
        output_path=out_file, 
        input_shape=(1, 3, 32, 32)
    )
    
    # Verify file was created
    assert Path(result_path).exists()
    assert os.path.getsize(result_path) > 0


def test_profile_latency():
    """Test the latency profiling utility."""
    model = DummyModel()
    
    # Profile on CPU with a small number of runs
    results = profile_latency(
        model, 
        input_shape=(1, 3, 32, 32), 
        num_warmup=2, 
        num_runs=5, 
        device="cpu"
    )
    
    # Verify results dictionary
    assert "mean_latency_ms" in results
    assert "std_latency_ms" in results
    assert "throughput_fps" in results
    
    assert results["mean_latency_ms"] > 0
    assert results["throughput_fps"] > 0
