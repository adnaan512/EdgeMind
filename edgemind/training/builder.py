"""
Optimizer and Scheduler builder for EdgeMind AI.

Design Philosophy:
    Instantiating optimizers directly tightly couples the training loop to 
    specific hyperparameter choices. Instead, this builder reads the YAML 
    config and instantiates the requested PyTorch optimizer and scheduler, 
    allowing rapid experimentation (e.g., swapping Adam for SGD via config).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler

from edgemind.core.logger import get_logger

logger = get_logger("training.builder")


def build_optimizer(
    model: nn.Module,
    config: Any
) -> optim.Optimizer:
    """Build PyTorch optimizer from config.
    
    Args:
        model: The neural network model.
        config: The EdgeMindConfig object containing the 'training' section.
        
    Returns:
        Configured PyTorch Optimizer.
    """
    train_cfg = config.get("training", {})
    
    name = str(train_cfg.get("optimizer", "adam")).lower()
    lr = float(train_cfg.get("learning_rate", 0.001))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    
    # Only pass parameters that require gradients (vital for transfer learning)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if not trainable_params:
        raise ValueError("Model has no trainable parameters. Did you freeze everything?")

    if name == "adam":
        optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    elif name == "adamw":
        optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        momentum = float(train_cfg.get("momentum", 0.9))
        optimizer = optim.SGD(
            trainable_params, 
            lr=lr, 
            momentum=momentum, 
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {name}. Use 'adam', 'adamw', or 'sgd'.")
        
    logger.info(f"Built Optimizer: {name.upper()} (lr={lr}, wd={weight_decay})")
    return optimizer


def build_scheduler(
    optimizer: optim.Optimizer,
    config: Any
) -> LRScheduler | None:
    """Build PyTorch learning rate scheduler from config.
    
    Args:
        optimizer: The optimizer to wrap.
        config: The EdgeMindConfig object containing the 'training' section.
        
    Returns:
        A PyTorch LRScheduler, or None if no scheduler is requested.
    """
    train_cfg = config.get("training", {})
    
    name = str(train_cfg.get("scheduler", "none")).lower()
    epochs = int(train_cfg.get("epochs", 10))
    
    if name == "none" or name == "null" or not name:
        return None
        
    if name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif name == "step":
        step_size = int(train_cfg.get("lr_step_size", 30))
        gamma = float(train_cfg.get("lr_gamma", 0.1))
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        raise ValueError(f"Unsupported scheduler: {name}. Use 'cosine', 'step', or 'none'.")
        
    logger.info(f"Built Scheduler: {name.capitalize()}")
    return scheduler
