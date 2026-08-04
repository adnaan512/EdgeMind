"""
Inference profiling utilities for EdgeMind AI.

Design Philosophy:
    Before deploying to edge hardware, researchers need to profile the 
    theoretical limits of their models. This module simulates edge 
    constraints by measuring CPU inference latency and throughput.
"""

from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn as nn

from edgemind.core.logger import get_logger

logger = get_logger("inference.profiler")


def profile_latency(
    model: nn.Module,
    input_shape: tuple[int, ...] = (1, 3, 224, 224),
    num_warmup: int = 10,
    num_runs: int = 100,
    device: str = "cpu"
) -> dict[str, float]:
    """Profile the inference latency of a PyTorch model.
    
    Simulates edge inference by measuring how long a single forward pass takes.
    
    Args:
        model: PyTorch model to profile.
        input_shape: Shape of the input tensor.
        num_warmup: Number of iterations to run before starting the timer (to warm up caches).
        num_runs: Number of iterations to run for timing.
        device: 'cpu' (simulates Edge CPU) or 'cuda'.
        
    Returns:
        Dictionary containing mean latency, standard deviation, and throughput.
    """
    model.eval()
    model = model.to(device)
    dummy_input = torch.randn(*input_shape, device=device)
    
    logger.info(f"Profiling latency on {device.upper()} with shape {input_shape}...")
    
    with torch.no_grad():
        # Warmup
        for _ in range(num_warmup):
            _ = model(dummy_input)
            
        if device == "cuda":
            torch.cuda.synchronize()
            
        # Timing runs
        latencies = []
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(dummy_input)
            
            if device == "cuda":
                torch.cuda.synchronize()
                
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)  # Convert to milliseconds
            
    latencies = np.array(latencies)
    mean_latency = float(np.mean(latencies))
    std_latency = float(np.std(latencies))
    
    # Calculate throughput (Frames Per Second) based on batch size and mean latency
    batch_size = input_shape[0]
    fps = (1000.0 / mean_latency) * batch_size
    
    logger.info(f"Mean Latency: {mean_latency:.2f} ms ± {std_latency:.2f} ms")
    logger.info(f"Throughput: {fps:.2f} FPS")
    
    return {
        "mean_latency_ms": mean_latency,
        "std_latency_ms": std_latency,
        "throughput_fps": fps
    }
