import os
import json
import time
from datetime import datetime
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import Config
from src.dataset import create_data_loaders
from src.model import LivenessNet


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def save_training_curves(
    train_losses: list,
    val_losses: list,
    train_accs: list,
    val_accs: list,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(train_losses, label="Train Loss", color="#2196F3", linewidth=2)
    axes[0].plot(val_losses, label="Val Loss", color="#FF5722", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(train_accs, label="Train Acc", color="#4CAF50", linewidth=2)
    axes[1].plot(val_accs, label="Val Acc", color="#FF9800", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Training curves saved to {path}")


def main() -> None:
    cfg = Config()
    device = torch.device(cfg.device_str)
    print(f"[INFO] Using device: {device}")

    os.makedirs("models", exist_ok=True)
    os.makedirs(cfg.output_dir, exist_ok=True)

    train_loader, val_loader, test_loader = create_data_loaders(cfg)
    print(f"[INFO] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    model = LivenessNet(pretrained=cfg.pretrained, dropout=cfg.dropout).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    train_losses: list = []
    val_losses: list = []
    train_accs: list = []
    val_accs: list = []

    for epoch in range(1, cfg.epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{cfg.epochs}")
        print(f"{'='*50}")

        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = validate(model, val_loader, criterion, device)

        train_losses.append(t_loss)
        val_losses.append(v_loss)
        train_accs.append(t_acc)
        val_accs.append(v_acc)

        print(f"  Train Loss: {t_loss:.4f} | Train Acc: {t_acc:.4f}")
        print(f"  Val   Loss: {v_loss:.4f} | Val   Acc: {v_acc:.4f}")

        scheduler.step(v_loss)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), cfg.model_save_path)
            print(f"  [SAVE] Best model saved (val_loss={v_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  [WAIT] No improvement ({patience_counter}/{cfg.patience})")

        if patience_counter >= cfg.patience:
            print(f"[INFO] Early stopping at epoch {epoch}")
            break

    save_training_curves(train_losses, val_losses, train_accs, val_accs, cfg.output_dir)

    meta = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "final_train_acc": train_accs[-1],
        "final_val_acc": val_accs[-1],
        "total_epochs": len(train_losses),
        "timestamp": datetime.now().isoformat(),
    }
    meta_path = os.path.join(cfg.output_dir, "training_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] Training metadata saved to {meta_path}")


if __name__ == "__main__":
    main()