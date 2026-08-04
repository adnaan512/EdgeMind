# Phase 5 Report — Model Compression & Optimization

**Date**: August 2026  
**Status**: ✅ Complete

---

## Objective

Implement industry-standard Edge AI optimization techniques (Magnitude Pruning and Dynamic Quantization) to shrink model size and prepare for hardware-constrained inference.

## What Was Built

### Core Modules

| Module | Purpose |
|---|---|
| `pruning.py` | Contains `apply_global_pruning` to zero out the smallest X% of weights globally across all Conv2d and Linear layers using PyTorch's native `prune` utility. Also includes logic to remove the pruning reparameterizations to make the zeros permanent for deployment. |
| `quantization.py` | Implements `apply_dynamic_quantization`, which converts FP32 Linear layer weights into INT8. Includes `get_model_size_mb` to prove the memory reduction. |
| `tools/compress.py` | A dedicated script that loads a trained `.pth` checkpoint, applies the requested percentage of pruning, dynamically quantizes it, and saves the highly compressed model ready for the edge. |

## Design Patterns Used

1. **Unstructured Global Pruning**: This is the most common baseline in model compression research. By pruning globally (rather than layer-by-layer), the algorithm naturally finds which layers have the most redundant weights.
2. **Dynamic Quantization**: Chosen for its simplicity and effectiveness. It requires no calibration data (unlike Static Quantization) while still providing a ~4x reduction in the memory footprint of Linear layers.

## Test Results

Unit tests confirmed that:
- Pruning successfully reaches the targeted sparsity (e.g., ~50% zeros).
- Pruning hooks are correctly removed, leaving behind a permanent static tensor.
- Dynamic INT8 quantization successfully reduces the model size by >50% on parameter-heavy models without breaking the forward pass.

```bash
python -m pytest tests/test_optimization.py -v
```

## Next Phase

**Phase 6: Inference Engine & Edge Simulation** — We will write the code to actually export these compressed models (e.g., to ONNX format) and profile their theoretical inference latency on CPU, simulating edge hardware constraints.
