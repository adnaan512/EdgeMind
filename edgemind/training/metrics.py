"""
Metrics tracking for EdgeMind AI.

Design Philosophy:
    Researchers need precise tracking of metrics (Loss, Accuracy) during 
    both training and validation phases. Instead of manual accumulation 
    loops, we use AverageMeters to keep running totals cleanly.
"""

from __future__ import annotations

import torch


class AverageMeter:
    """Computes and stores the average and current value of a metric.
    
    This is a standard research pattern (used in PyTorch examples) for 
    tracking loss and accuracy across batches.
    """

    def __init__(self, name: str, fmt: str = ":.4f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        """Reset all statistics."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """Update statistics.
        
        Args:
            val: The current value of the metric for the batch.
            n: The batch size (used for weighted averaging).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        fmtstr = f"{{name}} {{val{self.fmt}}} ({{avg{self.fmt}}})"
        return fmtstr.format(**self.__dict__)


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple = (1,)) -> list[float]:
    """Computes the accuracy over the k top predictions.
    
    Args:
        output: Model predictions of shape (batch_size, num_classes).
        target: Ground truth labels of shape (batch_size,).
        topk: Tuple of k values to compute top-k accuracy for.
        
    Returns:
        List of accuracy percentages [0.0, 100.0] corresponding to each k in topk.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Get the top k predictions
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        
        # Check if predictions match the target
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append((correct_k.mul_(100.0 / batch_size)).item())
            
        return res
