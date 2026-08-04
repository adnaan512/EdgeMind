"""
Core Training Engine for EdgeMind AI.

Design Philosophy:
    The Trainer class orchestrates the entire training loop. By encapsulating 
    this logic, we ensure:
    - Consistent execution across experiments
    - Proper device placement (GPU/MPS/CPU)
    - Automatic metric tracking and checkpointing
    - Cleaner top-level scripts
"""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from edgemind.core.device import get_device
from edgemind.core.logger import get_logger
from edgemind.training.builder import build_optimizer, build_scheduler
from edgemind.training.checkpoint import ModelCheckpoint
from edgemind.training.metrics import AverageMeter, accuracy

logger = get_logger("training.trainer")


class Trainer:
    """Orchestrates model training and validation."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Any,
    ):
        """Initialize the Trainer.
        
        Args:
            model: The neural network to train.
            train_loader: DataLoader for the training set.
            val_loader: DataLoader for the validation set.
            config: EdgeMindConfig containing training parameters.
        """
        self.config = config
        self.train_cfg = config.get("training", {})
        
        # Hardware
        preferred_device = config.get("device", {}).get("preferred", None)
        self.device = get_device(preferred_device)
        self.model = model.to(self.device)
        logger.info(f"Trainer initialized. Target device: {self.device}")
        
        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Optimization
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = build_optimizer(self.model, config)
        self.scheduler = build_scheduler(self.optimizer, config)
        
        # Checkpointing
        save_dir = config.get("paths", {}).get("checkpoint_dir", "checkpoints")
        self.checkpointer = ModelCheckpoint(save_dir=save_dir, mode="max", monitor="val_acc")
        
        self.epochs = int(self.train_cfg.get("epochs", 10))
        self.start_epoch = 1

    def fit(self) -> dict[str, float]:
        """Execute the full training loop.
        
        Returns:
            Dictionary containing the best validation metrics achieved.
        """
        logger.info(f"Starting training for {self.epochs} epochs.")
        
        for epoch in range(self.start_epoch, self.epochs + 1):
            logger.info(f"--- Epoch {epoch}/{self.epochs} ---")
            
            # 1. Train for one epoch
            train_loss, train_acc = self._train_epoch(epoch)
            
            # 2. Validate
            val_loss, val_acc = self._validate_epoch(epoch)
            
            # 3. Update Learning Rate
            if self.scheduler is not None:
                self.scheduler.step()
                
            # 4. Checkpoint
            # Convert config to dict for serialization
            cfg_dict = self.config.to_dict() if hasattr(self.config, "to_dict") else self.config
            self.checkpointer.save(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                metric_value=val_acc,
                config=cfg_dict
            )
            
            logger.info(
                f"Epoch {epoch} Summary - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )
            
        logger.info(f"Training complete. Best Validation Accuracy: {self.checkpointer.best_metric:.2f}%")
        return {"best_val_acc": self.checkpointer.best_metric}

    def _train_epoch(self, epoch: int) -> tuple[float, float]:
        """Run one epoch of training."""
        self.model.train()
        
        losses = AverageMeter("Loss")
        top1 = AverageMeter("Acc@1")
        
        start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            acc1 = accuracy(outputs, targets, topk=(1,))[0]
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1, inputs.size(0))
            
            # Log progress
            if batch_idx % 20 == 0:
                logger.debug(
                    f"Train Epoch: {epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "
                    f"Acc: {top1.val:.2f}% ({top1.avg:.2f}%)"
                )
                
        epoch_time = time.time() - start_time
        logger.info(f"Train Epoch {epoch} completed in {epoch_time:.2f}s")
        
        return losses.avg, top1.avg

    def _validate_epoch(self, epoch: int) -> tuple[float, float]:
        """Run validation on the holdout set."""
        self.model.eval()
        
        losses = AverageMeter("Loss")
        top1 = AverageMeter("Acc@1")
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                acc1 = accuracy(outputs, targets, topk=(1,))[0]
                losses.update(loss.item(), inputs.size(0))
                top1.update(acc1, inputs.size(0))
                
        return losses.avg, top1.avg
