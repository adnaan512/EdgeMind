# Phase 7 Report — Explainability & Visualization

**Date**: August 2026  
**Status**: ✅ Complete

---

## Objective

Provide transparency for black-box neural networks by implementing Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize which regions of an image influence the model's predictions.

## What Was Built

### Core Modules

| Module | Purpose |
|---|---|
| `gradcam.py` | Contains the `GradCAM` class. It uses PyTorch forward/backward hooks to intercept the feature maps (activations) and gradients of any targeted convolutional layer during the backward pass. It then computes a weighted combination to produce a normalized 2D heatmap. |
| `tools/visualize_cam.py` | A command-line script to run Grad-CAM on a specific image. It automatically detects the final Conv2d layer (or accepts a target layer name), runs the forward/backward passes, and uses OpenCV to generate an overlaid heatmap visualization. |

## Design Patterns Used

1. **PyTorch Hooks**: The most robust way to extract intermediate activations and gradients without modifying the underlying model's architecture. This ensures `GradCAM` works out-of-the-box with any standard `nn.Module`.
2. **Auto-layer Targeting**: The `visualize_cam.py` script includes a heuristic to automatically find the final Convolutional layer in the network if the user doesn't specify one. The final layer usually contains the highest-level spatial semantics, making it ideal for Grad-CAM.

## Test Results

Unit tests confirmed that:
- Heatmaps are correctly generated with matching spatial dimensions (e.g., 32x32 for CIFAR) and normalized strictly between `[0, 1]`.
- The overlay utility correctly combines the float32 heatmap with an RGB image to produce a valid `uint8` image suitable for saving.

```bash
python -m pytest tests/test_explainability.py -v
```

## Next Phase

**Phase 8: Research Report & Demo App** — The final phase! We will build a simple UI (e.g., Gradio or Streamlit) to interactively run the pipeline, allowing you to upload an image and instantly see the prediction, confidence, and Grad-CAM heatmap.
