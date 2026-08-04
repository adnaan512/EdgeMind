"""
ONNX export utilities for EdgeMind AI.

Design Philosophy:
    PyTorch is excellent for training, but edge devices (Raspberry Pi, 
    Jetson Nano, Coral TPU) require optimized deployment formats. ONNX 
    (Open Neural Network Exchange) is the industry standard for bridging 
    research (PyTorch) and production (TensorRT, CoreML, OpenVINO).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn

from edgemind.core.logger import get_logger

logger = get_logger("inference.export")


def export_to_onnx(
    model: nn.Module,
    output_path: str | Path,
    input_shape: tuple[int, ...] = (1, 3, 224, 224),
    device: str = "cpu"
) -> str:
    """Export a PyTorch model to ONNX format.
    
    Args:
        model: The trained PyTorch model.
        output_path: Where to save the .onnx file.
        input_shape: Tuple representing (batch_size, channels, height, width).
        device: Device to use for export ('cpu' or 'cuda').
        
    Returns:
        The absolute path to the exported ONNX file.
    """
    model.eval()
    
    # ONNX export requires a dummy input tensor matching the expected shape
    dummy_input = torch.randn(*input_shape, device=device)
    model = model.to(device)
    
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Exporting model to ONNX format: {out_path}")
    logger.info(f"Using dummy input shape: {input_shape}")
    
    try:
        torch.onnx.export(
            model,                         # model being run
            dummy_input,                   # model input
            str(out_path),                 # where to save the model
            export_params=True,            # store the trained parameter weights inside the model file
            opset_version=13,              # the ONNX version to export the model to (13 is stable for most edge runtimes)
            do_constant_folding=True,      # whether to execute constant folding for optimization
            input_names=['input'],         # the model's input names
            output_names=['output'],       # the model's output names
            dynamic_axes={                 # dynamic axes allow variable batch sizes at inference
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        logger.info(f"ONNX export successful: {out_path}")
        return str(out_path.absolute())
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        raise
