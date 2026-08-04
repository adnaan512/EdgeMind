#!/usr/bin/env python3
"""
Model compression entry point for EdgeMind AI.

Usage:
    python tools/compress.py --config configs/experiments/mobilenet_cifar10.yaml --checkpoint path/to/best.pth --prune 0.3 --quantize
"""

import argparse
import sys
from pathlib import Path

# Add project root to path so we can run this from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from edgemind.core.config import EdgeMindConfig
from edgemind.core.logger import get_logger
from edgemind.models import MODELS
from edgemind.training.checkpoint import ModelCheckpoint
from edgemind.optimization import (
    apply_global_pruning,
    remove_pruning_reparameterization,
    count_zero_weights,
    apply_dynamic_quantization,
    get_model_size_mb
)

logger = get_logger("compress_script")


def main():
    parser = argparse.ArgumentParser(description="Compress a trained EdgeMind AI model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained .pth weights")
    parser.add_argument("--prune", type=float, default=0.0, help="Fraction of weights to prune (0.0 to 1.0)")
    parser.add_argument("--quantize", action="store_true", help="Apply dynamic INT8 quantization")
    parser.add_argument("--output", type=str, default="compressed_model.pth", help="Output path")
    args = parser.parse_args()

    # 1. Load Config and Model
    logger.info(f"Loading config from {args.config}")
    config = EdgeMindConfig.from_yaml(args.config)
    
    model_config = config.get("model", {}).to_dict() if hasattr(config.get("model"), "to_dict") else config.get("model", {})
    model = MODELS.build(model_config)
    
    # 2. Load Weights
    logger.info(f"Loading weights from {args.checkpoint}")
    ModelCheckpoint.load(args.checkpoint, model, device=torch.device("cpu"))
    model.eval()
    
    original_size = get_model_size_mb(model)
    logger.info(f"Original Model Size: {original_size:.2f} MB")
    
    # 3. Pruning
    if args.prune > 0.0:
        model = apply_global_pruning(model, amount=args.prune)
        model = remove_pruning_reparameterization(model)
        zeros, total, sparsity = count_zero_weights(model)
        logger.info(f"Pruning complete. Sparsity: {sparsity:.2f}% ({zeros}/{total} zero weights)")
        
    # 4. Quantization
    if args.quantize:
        model = apply_dynamic_quantization(model)
        quant_size = get_model_size_mb(model)
        logger.info(f"Quantization complete. New Model Size: {quant_size:.2f} MB")
        
    # 5. Save compressed model
    # Note: For pruned models, we often save the state_dict directly.
    # For dynamically quantized PyTorch models, we can also use torch.save
    torch.save(model.state_dict(), args.output)
    logger.info(f"Compressed model saved to {args.output}")


if __name__ == "__main__":
    main()
