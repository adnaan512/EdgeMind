"""
Quantization utilities for EdgeMind AI.

Design Philosophy:
    Quantization converts FP32 (32-bit float) weights to INT8 (8-bit int).
    This reduces model size by 4x and dramatically speeds up inference on
    Edge CPUs (like ARM Cortex or Raspberry Pi).
    
    We implement Dynamic Quantization (easiest, good for Linear layers) and 
    Static Quantization (harder, good for CNNs, requires calibration).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.ao.quantization import quantize_dynamic

from edgemind.core.logger import get_logger

logger = get_logger("optimization.quantization")


def apply_dynamic_quantization(model: nn.Module) -> nn.Module:
    """Apply dynamic INT8 quantization to Linear layers.
    
    Dynamic quantization converts weights to INT8 ahead of time, but 
    activations are quantized dynamically during inference. It is highly 
    effective for models heavy on Linear/LSTM layers, but less so for CNNs.
    
    Args:
        model: The FP32 PyTorch model (must be on CPU).
        
    Returns:
        The INT8 dynamically quantized model.
    """
    # Ensure model is on CPU (PyTorch quantization requires CPU)
    model.eval()
    model.cpu()
    
    logger.info("Applying Dynamic INT8 Quantization to Linear layers...")
    
    quantized_model = quantize_dynamic(
        model,
        {nn.Linear},  # Only quantize Linear layers dynamically
        dtype=torch.qint8
    )
    
    return quantized_model


def get_model_size_mb(model: nn.Module) -> float:
    """Calculate the physical size of the model in Megabytes.
    
    Useful for comparing the size of FP32 vs INT8 quantized models.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Size in MB.
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb
