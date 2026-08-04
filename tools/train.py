#!/usr/bin/env python3
"""
Main training entry point for EdgeMind AI.

Usage:
    python tools/train.py --config configs/experiments/mobilenet_cifar10.yaml
"""

import argparse
import sys
from pathlib import Path

# Add project root to path so we can run this from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edgemind.core.config import EdgeMindConfig
from edgemind.core.logger import get_logger
from edgemind.data.dataloader import build_dataloaders
from edgemind.models import MODELS
from edgemind.training.trainer import Trainer

logger = get_logger("train_script")


def main():
    parser = argparse.ArgumentParser(description="Train an EdgeMind AI model")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to the experiment YAML config file"
    )
    args = parser.parse_args()

    logger.info(f"Loading configuration from {args.config}")
    
    # We assume 'configs/base.yaml' is always the base
    base_config_path = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"
    
    config = EdgeMindConfig.from_yaml(
        args.config,
        base_config_path=base_config_path if base_config_path.exists() else None
    )
    
    # 1. Setup Data
    logger.info("Setting up data pipeline...")
    train_loader, val_loader = build_dataloaders(config)
    
    # 2. Setup Model
    logger.info("Setting up model...")
    model_config = config.get("model", {}).to_dict() if hasattr(config.get("model"), "to_dict") else config.get("model", {})
    model = MODELS.build(model_config)
    
    # 3. Setup Trainer
    logger.info("Setting up training engine...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    
    # 4. Execute
    trainer.fit()


if __name__ == "__main__":
    main()
