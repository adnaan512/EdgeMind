#!/usr/bin/env python3
"""
Grad-CAM visualization tool for EdgeMind AI.

Usage:
    python tools/visualize_cam.py --config configs/experiments/mobilenet_cifar10.yaml --checkpoint path/to/best.pth --image path/to/image.jpg
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edgemind.core.config import EdgeMindConfig
from edgemind.core.logger import get_logger
from edgemind.models import MODELS
from edgemind.training.checkpoint import ModelCheckpoint
from edgemind.explainability import GradCAM

logger = get_logger("visualize_cam")


def load_image(image_path: str, img_size: int = 32) -> tuple[torch.Tensor, np.ndarray]:
    """Load and preprocess an image for inference and visualization."""
    # 1. Load for visualization (RGB, float [0, 1])
    img_pil = Image.open(image_path).convert("RGB")
    img_resized = img_pil.resize((img_size, img_size))
    img_np = np.array(img_resized, dtype=np.float32) / 255.0
    
    # 2. Preprocess for model
    # Use standard ImageNet or CIFAR stats depending on your config.
    # We'll use standard CIFAR stats here as fallback.
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])
    
    img_tensor = transform(img_pil).unsqueeze(0)  # Add batch dimension
    return img_tensor, img_np


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmaps for an image.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained .pth weights")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="gradcam_result.jpg", help="Output image path")
    parser.add_argument("--target_layer", type=str, default=None, 
                        help="Name of the target Conv2d layer (e.g., 'backbone.features.18'). "
                             "If None, attempts to find the last Conv2d layer.")
    args = parser.parse_args()

    # 1. Load Config and Model
    logger.info(f"Loading config from {args.config}")
    config = EdgeMindConfig.from_yaml(args.config)
    
    model_config = config.get("model", {}).to_dict() if hasattr(config.get("model"), "to_dict") else config.get("model", {})
    model = MODELS.build(model_config)
    
    # 2. Load Weights
    logger.info(f"Loading weights from {args.checkpoint}")
    ModelCheckpoint.load(args.checkpoint, model, device=torch.device("cpu"))
    model.eval()
    
    # 3. Load Image
    img_size = config.get("data", {}).get("image_size", 32)
    img_tensor, img_np = load_image(args.image, img_size)
    
    # 4. Find Target Layer
    target_layer = None
    if args.target_layer:
        # Traverse the model to find the layer
        curr = model
        for part in args.target_layer.split('.'):
            curr = getattr(curr, part)
        target_layer = curr
    else:
        # Heuristic: Find the last Conv2d layer
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                logger.info(f"Auto-selected target layer: {name}")
                break
                
    if target_layer is None:
        raise ValueError("Could not find a target Conv2d layer for Grad-CAM.")
        
    # 5. Run Grad-CAM
    cam = GradCAM(model, target_layer)
    heatmap = cam(img_tensor)
    
    # 6. Overlay and Save
    overlay = GradCAM.overlay_heatmap(img_np, heatmap)
    
    # OpenCV expects BGR for saving
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, overlay_bgr)
    logger.info(f"Grad-CAM visualization saved to {args.output}")


if __name__ == "__main__":
    main()
