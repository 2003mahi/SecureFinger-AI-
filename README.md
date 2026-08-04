# SecureFinger AI — Contactless Fingerprint Anti-Spoofing System

A deep learning-based Presentation Attack Detection (PAD) system that classifies fingerprint images as **LIVE** or **SPOOF** to prevent fake fingerprint authentication attacks.

## Architecture

- **Backbone:** MobileNetV3-Small (pretrained on ImageNet)
- **Head:** AdaptiveAvgPool2d → Dropout → Linear(1024, 2)
- **Optimizer:** Adam (lr=0.001, weight_decay=1e-4)
- **Loss:** CrossEntropyLoss
- **Threshold Calibration:** APCER/BPCER sweep → BPCER ≈ 3%

## Project Structure

```
SecureFinger/
├── config.py                  # Hyperparameters & configuration
├── requirements.txt           # Python dependencies
├── liveness_train.py          # Training pipeline
├── liveness_eval.py           # Evaluation & threshold calibration
├── liveness_infer.py          # Single-image / webcam inference
├── app.py                     # Streamlit demo application
├── src/
│   ├── __init__.py
│   ├── dataset.py             # FingerprintDataset + data loaders
│   ├── model.py               # LivenessNet (MobileNetV3-Small)
│   └── gradcam.py             # Grad-CAM explainability
├── data/
│   ├── live/                  # LIVE fingerprint images
│   └── spoof/                 # SPOOF fingerprint images
├── models/
│   └── liveness_model.pth     # Saved best model weights
├── outputs/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── apcer_bpcer_curve.png
│   ├── score_distribution.png
│   ├── gradcam_*.png
│   └── evaluation_results.json
└── report.pdf                 # Technical report with 7 evaluation answers
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare data
Place fingerprint images in `data/live/` and `data/spoof/` (PNG/JPG/BMP).

### 3. Train the model
```bash
python liveness_train.py
```

### 4. Evaluate & generate plots
```bash
python liveness_eval.py
```

### 5. Run inference on a single image
```bash
python liveness_infer.py path/to/fingerprint.png
```

### 6. Launch the Streamlit demo
```bash
streamlit run app.py
```

## Key Features

- **Transfer Learning:** MobileNetV3-Small pretrained backbone with frozen feature extractor
- **Data Augmentation:** Rotation, brightness variation, horizontal flip, Gaussian noise
- **Threshold Calibration:** Sweeps 0–1 to find threshold where BPCER ≈ 3%
- **Grad-CAM Explainability:** Visualizes which fingerprint regions the model attends to
- **Full Evaluation Suite:** Confusion matrix, ROC curve, APCER/BPCER curve, score distribution

## Model Performance Metrics

| Metric | Description |
|--------|-------------|
| APCER | Attack Presentation Classification Error Rate |
| BPCER | Bona Fide Presentation Classification Error Rate |
| EER | Equal Error Rate (APCER = BPCER) |
| Threshold | Decision boundary calibrated to BPCER ≈ 3% |