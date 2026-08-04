# Phase 2 Report — Data Pipeline

**Date**: August 2026  
**Status**: ✅ Complete

---

## Objective

Build a modular, config-driven data pipeline capable of handling both standard benchmarks (CIFAR-10) and real-world custom image datasets.

## What Was Built

### Core Modules

| Module | Purpose |
|---|---|
| `transforms.py` | Factory for config-driven `torchvision` augmentations. Separates training vs val augmentations. |
| `base_dataset.py` | Abstract base class enforcing a common interface across all dataset implementations. |
| `cifar_dataset.py` | Wrapper for CIFAR-10, the standard benchmark for model compression and Edge AI. |
| `folder_dataset.py` | Custom dataset wrapper allowing the framework to train on any folder of images (e.g. manufacturing defects). |
| `dataloader.py` | Factory function that reads YAML configs, builds the transforms, instantiates the datasets via the Registry, and returns ready-to-use PyTorch DataLoaders. |
| `visualize.py` | Data inspection utility to unnormalize and plot a grid of augmented images. |
| `configs/data/cifar10.yaml` | Experiment configuration template for CIFAR-10. |

## Design Patterns Used

1. **Config-Driven Data Loading**: The `dataloader.py` factory reads settings (batch size, num_workers, dataset name) straight from the YAML config.
2. **Registry Integration**: Both `cifar10` and `custom_folder` datasets are registered in the `DATASETS` registry, allowing us to swap datasets simply by changing `data.dataset: "custom_folder"` in the config.

## Test Results

All new tests for transforms and data loading passed successfully:
```bash
python -m pytest tests/ -v
```

## Next Phase

**Phase 3: Model Zoo & Transfer Learning** — Implementing a registry of pretrained backbones (MobileNetV2, ResNet18, EfficientNet) and the base model wrapper for transfer learning.
