import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import time

from edgemind.models import MODELS
from edgemind.core.config import EdgeMindConfig
from edgemind.explainability import GradCAM

# CIFAR-10 class labels for demonstration
CIFAR_CLASSES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

@st.cache_resource
def load_model(config_path: str, checkpoint_path: str | None = None):
    config = EdgeMindConfig.from_yaml(config_path)
    model_config = config.get("model", {}).to_dict() if hasattr(config.get("model"), "to_dict") else config.get("model", {})
    model = MODELS.build(model_config)
    
    if checkpoint_path:
        try:
            # Fallback to load state dict safely for the demo if checkpoint doesn't exist yet
            from edgemind.training.checkpoint import ModelCheckpoint
            ModelCheckpoint.load(checkpoint_path, model, device=torch.device("cpu"))
        except Exception as e:
            st.warning(f"Could not load checkpoint ({e}). Using untrained weights for demonstration.")
            
    model.eval()
    return model

def process_image(img_pil: Image.Image, img_size: int = 32):
    img_resized = img_pil.resize((img_size, img_size))
    img_np = np.array(img_resized, dtype=np.float32) / 255.0
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])
    
    img_tensor = transform(img_pil).unsqueeze(0)
    return img_tensor, img_np

def get_target_layer(model):
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, torch.nn.Conv2d):
            return module
    return None

def main():
    st.set_page_config(page_title="EdgeMind AI Demo", layout="wide", initial_sidebar_state="expanded")
    
    st.title("EdgeMind AI: Edge AI Research Framework")
    st.markdown("Interactive demonstration of the EdgeMind AI classification and explainability pipeline.")
    
    st.sidebar.header("Configuration")
    config_path = st.sidebar.text_input("Config YAML", value="configs/experiments/mobilenet_cifar10.yaml")
    checkpoint_path = st.sidebar.text_input("Checkpoint Path", value="")
    
    # Load model
    try:
        model = load_model(config_path, checkpoint_path if checkpoint_path else None)
        st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")
        st.stop()
        
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        img_pil = Image.open(uploaded_file).convert("RGB")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Input Image")
            st.image(img_pil, use_container_width=True)
            
        with st.spinner("Processing..."):
            img_tensor, img_np = process_image(img_pil)
            
            # 1. Inference and Latency
            start_time = time.time()
            with torch.no_grad():
                output = model(img_tensor)
            latency_ms = (time.time() - start_time) * 1000
            
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            class_idx = torch.argmax(probabilities).item()
            confidence = probabilities[class_idx].item()
            
            # Map to class name (assuming CIFAR-10)
            class_name = CIFAR_CLASSES[class_idx] if class_idx < len(CIFAR_CLASSES) else f"Class {class_idx}"
            
            with col2:
                st.subheader("Prediction Results")
                st.metric("Predicted Class", class_name)
                st.metric("Confidence", f"{confidence*100:.2f}%")
                st.metric("CPU Latency", f"{latency_ms:.2f} ms")
                
            # 2. Grad-CAM Explainability
            target_layer = get_target_layer(model)
            if target_layer:
                cam = GradCAM(model, target_layer)
                heatmap = cam(img_tensor, class_idx=class_idx)
                overlay = GradCAM.overlay_heatmap(img_np, heatmap)
                
                with col3:
                    st.subheader("Grad-CAM Explainability")
                    st.image(overlay, use_container_width=True, channels="RGB")
                    st.caption(f"Target Layer: {target_layer.__class__.__name__}")
            else:
                with col3:
                    st.warning("No Conv2d layer found for Grad-CAM.")

if __name__ == "__main__":
    main()
