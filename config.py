from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    # ── Data ──────────────────────────────────────────────
    data_root: str = "data"
    live_dir: str = "data/live"
    spoof_dir: str = "data/spoof"
    image_size: int = 224
    batch_size: int = 16
    num_workers: int = 0

    # ── Split ratios ──────────────────────────────────────
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # ── Augmentation ──────────────────────────────────────
    rotation_range: float = 15.0
    brightness_range: float = 0.2
    gaussian_noise_std: float = 0.05
    horizontal_flip_prob: float = 0.5

    # ── Model ─────────────────────────────────────────────
    pretrained: bool = True
    num_classes: int = 2
    dropout: float = 0.3

    # ── Training ──────────────────────────────────────────
    epochs: int = 25
    lr: float = 0.001
    weight_decay: float = 1e-4
    patience: int = 5
    device: str = "auto"

    # ── Paths ─────────────────────────────────────────────
    model_save_path: str = "models/liveness_model.pth"
    output_dir: str = "outputs"

    # ── Threshold calibration ─────────────────────────────
    target_bpcER: float = 0.03
    threshold_steps: int = 200

    # ── Inference ─────────────────────────────────────────
    inference_batch_size: int = 8

    @property
    def device_str(self) -> str:
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


def get_config() -> Config:
    return Config()