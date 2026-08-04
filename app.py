import os
import json
import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

from config import Config
from src.model import LivenessNet
from src.gradcam import compute_gradcam, overlay_gradcam


st.set_page_config(page_title="SecureFinger AI", page_icon="🖐", layout="wide")

st.title("🖐 SecureFinger AI — Fingerprint Liveness Detection")
st.markdown("Contactless Fingerprint Presentation Attack Detection System")


@st.cache_resource
def load_model(model_path: str):
    cfg = Config()
    device = torch.device(cfg.device_str)
    model = LivenessNet(pretrained=False, dropout=cfg.dropout)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device


model, device = load_model(Config().model_save_path)

uploaded_file = st.file_uploader(
    "Upload a fingerprint image",
    type=["png", "jpg", "jpeg", "bmp"],
    help="Upload a fingerprint image for live/spoof classification",
)

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, caption="Input Fingerprint", use_container_width=True)

    if image.mode == 'L':
        image = image.convert('RGB')
    elif image.mode == 'RGBA':
        image = image.convert('RGB')

    cfg = Config()
    preprocess = T.Compose([
        T.Resize((cfg.image_size, cfg.image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, dim=1)

    pred_label = ["LIVE", "SPOOF"][predicted.item()]
    confidence_val = confidence.item()

    with col2:
        st.subheader("Prediction Result")
        if pred_label == "LIVE":
            st.success(f"**{pred_label}**")
        else:
            st.error(f"**{pred_label}**")

        st.metric("Confidence", f"{confidence_val:.2%}")

        if pred_label == "LIVE":
            st.markdown("✅ **Security Recommendation:** Fingerprint appears genuine. Authentication can proceed.")
        else:
            st.markdown("⚠️ **Security Recommendation:** Spoof fingerprint detected! Reject authentication attempt.")

    st.subheader("Grad-CAM Heatmap")
    heatmap = compute_gradcam(model, img_tensor, target_class=1)
    overlay = overlay_gradcam(np.array(image), heatmap, alpha=0.5)

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.image(heatmap, caption="Attention Heatmap", use_container_width=True)
    with col_h2:
        st.image(overlay, caption="Overlay on Original", use_container_width=True)
