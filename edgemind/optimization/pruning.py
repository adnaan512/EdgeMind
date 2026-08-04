"""
Magnitude-based unstructured pruning for EdgeMind AI.

Design Philosophy:
    Pruning removes unimportant weights (setting them to zero) to reduce 
    model size and theoretical FLOPs. For Edge AI research, global magnitude
    pruning is a standard baseline: it looks at all convolutional and linear 
    weights across the entire model and zeroes out the smallest X%.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from edgemind.core.logger import get_logger

logger = get_logger("optimization.pruning")


def apply_global_pruning(model: nn.Module, amount: float = 0.3) -> nn.Module:
    """Apply global unstructured magnitude pruning to the model.
    
    This function finds all Conv2d and Linear layers and prunes the specified
    percentage of the smallest weights globally across all those layers.
    
    Args:
        model: The PyTorch model to prune.
        amount: Float between 0.0 and 1.0 indicating the fraction of weights to prune.
        
    Returns:
        The pruned model (modifies in-place, but returns for convenience).
    """
    if not (0.0 <= amount < 1.0):
        raise ValueError(f"Pruning amount must be between 0.0 and 1.0, got {amount}")
        
    parameters_to_prune = []
    
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))
            
    if not parameters_to_prune:
        logger.warning("No Conv2d or Linear layers found to prune.")
        return model
        
    logger.info(f"Applying global magnitude pruning (amount={amount*100:.1f}%)")
    
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    
    return model


def remove_pruning_reparameterization(model: nn.Module) -> nn.Module:
    """Remove the pruning reparameterization to make the zeros permanent.
    
    PyTorch's prune module creates a 'weight_orig' and a 'weight_mask'.
    To actually deploy the model, we must remove this hook so that 'weight'
    is just a standard tensor with zeros in it.
    
    Args:
        model: The pruned PyTorch model.
        
    Returns:
        The model with pruning made permanent.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            try:
                prune.remove(module, 'weight')
                count += 1
            except ValueError:
                # Module was not pruned
                pass
                
    logger.info(f"Removed pruning reparameterization from {count} layers.")
    return model


def count_zero_weights(model: nn.Module) -> tuple[int, int, float]:
    """Calculate the sparsity of the model.
    
    Returns:
        Tuple of (zero_weights, total_weights, sparsity_percentage).
    """
    zeros = 0
    total = 0
    
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, "weight") and module.weight is not None:
                zeros += float(torch.sum(module.weight == 0))
                total += float(module.weight.nelement())
                
    if total == 0:
        return 0, 0, 0.0
        
    sparsity = (zeros / total) * 100
    return int(zeros), int(total), sparsity
