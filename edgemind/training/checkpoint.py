"""
Model checkpointing system for EdgeMind AI.

Design Philosophy:
    Training can crash, or models can overfit. A reliable checkpointing 
    system saves the model state periodically and explicitly tracks the 
    "best" weights based on a validation metric (like accuracy).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from edgemind.core.logger import get_logger

logger = get_logger("training.checkpoint")


class ModelCheckpoint:
    """Saves model checkpoints during training.
    
    Automatically saves the most recent epoch and maintains a separate copy 
    of the "best" model based on validation performance.
    """

    def __init__(
        self,
        save_dir: str | Path,
        mode: str = "max",
        monitor: str = "val_acc"
    ):
        """Initialize the checkpointer.
        
        Args:
            save_dir: Directory where weights will be saved.
            mode: "max" (e.g., accuracy) or "min" (e.g., loss).
            monitor: The name of the metric being tracked.
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.mode = mode
        self.monitor = monitor
        
        # Track the best metric value
        self.best_metric = float('-inf') if mode == "max" else float('inf')

    def is_better(self, current: float, best: float) -> bool:
        """Check if the current metric is better than the best recorded."""
        if self.mode == "max":
            return current > best
        return current < best

    def save(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        metric_value: float,
        config: dict[str, Any]
    ) -> None:
        """Save a checkpoint.
        
        Always saves 'last.pth'. If metric_value is the best seen so far, 
        also saves 'best.pth'.
        """
        state_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            self.monitor: metric_value,
            "config": config,
        }
        
        last_path = self.save_dir / "last.pth"
        torch.save(state_dict, last_path)
        
        if self.is_better(metric_value, self.best_metric):
            logger.info(
                f"Metric '{self.monitor}' improved from {self.best_metric:.4f} "
                f"to {metric_value:.4f}. Saving best model."
            )
            self.best_metric = metric_value
            best_path = self.save_dir / "best.pth"
            shutil.copyfile(last_path, best_path)

    @staticmethod
    def load(
        checkpoint_path: str | Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        device: torch.device | None = None
    ) -> tuple[int, float]:
        """Load a model state from a checkpoint.
        
        Returns:
            A tuple of (epoch, metric_value).
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {path}")
            
        if device is None:
            device = torch.device("cpu")
            
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
        # Extract epoch and the main metric (we guess it's the 4th key usually)
        epoch = checkpoint.get("epoch", 0)
        
        # Try to find the metric value
        metric_val = 0.0
        for key, val in checkpoint.items():
            if key not in ["epoch", "model_state_dict", "optimizer_state_dict", "config"]:
                metric_val = val
                break
                
        logger.info(f"Loaded checkpoint '{path}' (epoch {epoch})")
        return epoch, metric_val
