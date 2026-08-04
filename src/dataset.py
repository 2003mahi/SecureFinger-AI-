import os
import random
from typing import Dict, List, Tuple

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from PIL import Image

from config import Config


class FingerprintDataset(Dataset):
    """Dataset that loads fingerprint images from live/ and spoof/ subdirectories.

    Class mapping:
        0 -> LIVE
        1 -> SPOOF
    """

    CLASS_NAMES: List[str] = ["LIVE", "SPOOF"]
    CLASS_TO_IDX: Dict[str, int] = {"LIVE": 0, "SPOOF": 1}

    def __init__(
        self,
        live_dir: str,
        spoof_dir: str,
        image_size: int = 224,
        augment: bool = False,
        config: Config | None = None,
    ) -> None:
        self.samples: List[Tuple[str, int]] = []

        for img_file in sorted(os.listdir(live_dir)):
            if img_file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                self.samples.append((os.path.join(live_dir, img_file), 0))

        for img_file in sorted(os.listdir(spoof_dir)):
            if img_file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                self.samples.append((os.path.join(spoof_dir, img_file), 1))

        if len(self.samples) == 0:
            raise ValueError("No images found in live/ and spoof/ directories.")

        self.image_size = image_size
        self.augment = augment
        self.cfg = config or Config()

        self.transform = self._build_transforms()

    def _build_transforms(self) -> transforms.Compose:
        base = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
        return transforms.Compose(base)

    def _augment_image(self, img: np.ndarray) -> np.ndarray:
        if random.random() < self.cfg.horizontal_flip_prob:
            img = cv2.flip(img, 1)

        angle = random.uniform(-self.cfg.rotation_range, self.cfg.rotation_range)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

        brightness_factor = 1.0 + random.uniform(
            -self.cfg.brightness_range, self.cfg.brightness_range
        )
        img = cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)

        noise = np.random.normal(
            0, self.cfg.gaussian_noise_std * 255, img.shape
        ).astype(np.float32)
        img = cv2.add(img.astype(np.float32), noise)
        img = np.clip(img, 0, 255).astype(np.uint8)

        return img

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        path, label = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Failed to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.augment:
            img = self._augment_image(img)

        img_pil = Image.fromarray(img)
        tensor = self.transform(img_pil)
        return tensor, label


def create_datasets(config: Config) -> Tuple:
    """Create train / validation / test splits from the data directories.

    Returns:
        train_ds, val_ds, test_ds
    """
    full_ds = FingerprintDataset(
        live_dir=config.live_dir,
        spoof_dir=config.spoof_dir,
        image_size=config.image_size,
        augment=False,
        config=config,
    )

    n = len(full_ds)
    n_train = int(n * config.train_ratio)
    n_val = int(n * config.val_ratio)
    n_test = n - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    train_ds.dataset.augment = True

    return train_ds, val_ds, test_ds


def create_data_loaders(config: Config):
    """Create DataLoader objects for train, val, and test splits."""
    import torch

    train_ds, val_ds, test_ds = create_datasets(config)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
