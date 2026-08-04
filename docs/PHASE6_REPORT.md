# Phase 6 Report — Inference Engine & Edge Simulation

**Date**: August 2026  
**Status**: ✅ Complete

---

## Objective

Bridge the gap between model training and hardware deployment by implementing an ONNX exporter and an inference profiler that simulates edge CPU constraints.

## What Was Built

### Core Modules

| Module | Purpose |
|---|---|
| `export.py` | Implements `export_to_onnx`. This leverages `torch.onnx.export` (opset 13) to convert dynamic PyTorch graphs into a static, highly optimizable `.onnx` file format used by deployment engines like TensorRT, CoreML, and OpenVINO. |
| `profiler.py` | Implements `profile_latency`. Before deploying to a physical Raspberry Pi or Jetson Nano, researchers need to know if a model architecture is even viable. This function runs a warm-up phase, measures the precise CPU inference time across hundreds of runs, and outputs mean latency and Frames Per Second (FPS). |
| `tools/export_onnx.py` | A command-line script that loads a trained `.pth` checkpoint, optionally runs the CPU profiler to gauge FPS, and exports the model to `.onnx`. |
| `requirements.txt` | Added a standard Python requirements file to track `onnx`, `torch`, `pytest`, etc. |

## Design Patterns Used

1. **Hardware Agnosticism via ONNX**: By exporting to ONNX, we decouple our training framework (PyTorch) from the eventual deployment hardware. This is a massive selling point for a scalable research project.
2. **Dynamic Batching**: The ONNX export is configured with `dynamic_axes={'input': {0: 'batch_size'}}`, meaning the exported model can handle a batch size of 1 for real-time edge streaming, or 32 for batched server inference.

## Test Results

Unit tests confirm that:
- `.onnx` files are generated correctly and contain valid protobuf graph structures.
- The profiler successfully calculates and returns latency (ms) and throughput (FPS).

```bash
python -m pytest tests/test_inference.py -v
```

## Next Phase

**Phase 7: Explainability & Visualization** — Edge AI models are often black boxes. We will implement Grad-CAM to generate heatmaps showing *where* the model is looking when it makes a decision, which is crucial for medical or industrial AI deployments.
