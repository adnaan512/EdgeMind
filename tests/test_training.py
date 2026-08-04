"""Tests for training engine components."""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from edgemind.core.config import EdgeMindConfig
from edgemind.training.builder import build_optimizer, build_scheduler
from edgemind.training.checkpoint import ModelCheckpoint
from edgemind.training.metrics import AverageMeter, accuracy


def test_average_meter():
    """Test metric accumulation."""
    meter = AverageMeter("loss")
    
    meter.update(2.0, n=1)
    assert meter.avg == 2.0
    
    meter.update(4.0, n=3)
    # Total sum: 2.0*1 + 4.0*3 = 14.0. Total count: 4. Avg: 3.5
    assert meter.avg == 3.5
    assert meter.sum == 14.0


def test_accuracy():
    """Test top-k accuracy calculation."""
    # Batch of 4 predictions for 3 classes
    output = torch.tensor([
        [0.8, 0.1, 0.1], # Pred: 0
        [0.2, 0.6, 0.2], # Pred: 1
        [0.1, 0.1, 0.8], # Pred: 2
        [0.6, 0.3, 0.1], # Pred: 0
    ])
    target = torch.tensor([0, 1, 1, 2]) # 0 and 1 are correct, others wrong.
    
    # 2 out of 4 correct = 50%
    acc = accuracy(output, target, topk=(1,))
    assert acc[0] == 50.0


def test_build_optimizer():
    """Test optimizer builder filters frozen params."""
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.Linear(10, 2)
    )
    
    # Freeze first layer
    for param in model[0].parameters():
        param.requires_grad = False
        
    config = EdgeMindConfig.from_dict({
        "training": {
            "optimizer": "adam",
            "learning_rate": 0.01
        }
    })
    
    optimizer = build_optimizer(model, config)
    assert isinstance(optimizer, torch.optim.Adam)
    
    # Should only contain parameters for the second layer (weight and bias)
    assert len(optimizer.param_groups[0]["params"]) == 2


def test_model_checkpoint(tmp_path):
    """Test saving and loading the best model."""
    checkpointer = ModelCheckpoint(save_dir=tmp_path, mode="max", monitor="acc")
    
    model = nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    # Epoch 1: 50% accuracy
    checkpointer.save(1, model, optimizer, 50.0, {})
    assert (tmp_path / "last.pth").exists()
    assert (tmp_path / "best.pth").exists()
    
    # Epoch 2: 40% accuracy (worse)
    checkpointer.save(2, model, optimizer, 40.0, {})
    
    # Epoch 3: 60% accuracy (better)
    checkpointer.save(3, model, optimizer, 60.0, {})
    
    # Load best
    epoch, metric = ModelCheckpoint.load(tmp_path / "best.pth", model, optimizer)
    assert epoch == 3
    assert metric == 60.0
