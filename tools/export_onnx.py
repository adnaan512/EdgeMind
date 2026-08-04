#!/usr/bin/env python3
"""
ONNX Export and Profiling entry point for EdgeMind AI.

Usage:
    python tools/export_onnx.py --config configs/experiments/mobilenet_cifar10.yaml --checkpoint path/to/best.pth --output model.onnx --profile
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from edgemind.core.config import EdgeMindConfig
from edgemind.core.logger import get_logger
from edgemind.models import MODELS
from edgemind.training.checkpoint import ModelCheckpoint
from edgemind.inference import export_to_onnx, profile_latency

logger = get_logger("export_script")


def main():
    parser = argparse.ArgumentParser(description="Export an EdgeMind AI model to ONNX and profile it.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained .pth weights")
    parser.add_argument("--output", type=str, default="model.onnx", help="Output ONNX path")
    parser.add_argument("--profile", action="store_true", help="Profile CPU inference latency")
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
    
    # Extract input shape from config (fallback to 32x32 CIFAR size)
    data_cfg = config.get("data", {})
    img_size = data_cfg.get("image_size", 32)
    input_shape = (1, 3, img_size, img_size)
    
    # 3. Profile (Optional)
    if args.profile:
        logger.info("Starting CPU inference profiling...")
        profile_latency(model, input_shape=input_shape, num_runs=100, device="cpu")
        
    # 4. Export
    export_to_onnx(model, output_path=args.output, input_shape=input_shape, device="cpu")


if __name__ == "__main__":
    main()
