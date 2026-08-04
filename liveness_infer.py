import json
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from config import Config
from src.model import LivenessNet
from src.gradcam import compute_gradcam, overlay_gradcam


class InferenceEngine:
    """Runs inference on a single fingerprint image or a webcam feed."""

    def __init__(self, model_path: str, config: Config | None = None) -> None:
        self.cfg = config or Config()
        self.device = torch.device(self.cfg.device_str)

        self.model = LivenessNet(pretrained=False, dropout=self.cfg.dropout)
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((self.cfg.image_size, self.cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def predict(self, image_path: str) -> dict:
        """Classify a single fingerprint image as LIVE or SPOOF.

        Returns:
            dict with keys: prediction, confidence, gradcam_path (optional)
        """
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, dim=1)

        pred_label = ["LIVE", "SPOOF"][predicted.item()]
        confidence_val = confidence.item()

        result = {
            "prediction": pred_label,
            "confidence": round(confidence_val, 4),
        }
        return result

    def predict_with_gradcam(
        self,
        image_path: str,
        output_dir: str = "outputs",
    ) -> dict:
        """Run prediction and generate a Grad-CAM heatmap overlay."""
        import os

        result = self.predict(image_path)

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

        heatmap = compute_gradcam(self.model, tensor, target_class=1)

        orig_resized = cv2.resize(img_rgb, (224, 224))
        overlay = overlay_gradcam(orig_resized, heatmap, alpha=0.5)

        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        heatmap_path = os.path.join(output_dir, f"{base_name}_gradcam.png")
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(heatmap_path, overlay_bgr)

        result["gradcam_path"] = heatmap_path
        return result

    def predict_webcam(self, camera_index: int = 0, max_frames: int = 1) -> dict:
        """Capture a frame from the webcam and classify it."""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")

        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError("Failed to capture frame from webcam")

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, dim=1)

        pred_label = ["LIVE", "SPOOF"][predicted.item()]
        confidence_val = confidence.item()

        return {
            "prediction": pred_label,
            "confidence": round(confidence_val, 4),
        }


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python liveness_infer.py <image_path> [model_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else Config().model_save_path

    engine = InferenceEngine(model_path)
    result = engine.predict_with_gradcam(image_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()