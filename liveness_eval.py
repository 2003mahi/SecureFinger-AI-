import os
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    classification_report,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import Config
from src.dataset import FingerprintDataset, create_data_loaders
from src.model import LivenessNet
from src.gradcam import compute_gradcam, overlay_gradcam


def compute_apcer_bpcer(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> Tuple[float, float]:
    """Compute APCER and BPCER for a given threshold.

    APCER = Attack Presentation Classification Error Rate
        (fraction of SPOOF samples classified as LIVE)
    BPCER = Bona Fide Presentation Classification Error Rate
        (fraction of LIVE samples classified as SPOOF)
    """
    predictions = (scores >= threshold).astype(int)

    live_mask = labels == 0
    spoof_mask = labels == 1

    apcer = np.mean(predictions[spoof_mask] == 0) if spoof_mask.any() else 0.0
    bpcer = np.mean(predictions[live_mask] == 1) if live_mask.any() else 0.0

    return float(apcer), float(bpcer)


def find_threshold_at_target_bpcer(
    scores: np.ndarray,
    labels: np.ndarray,
    target_bpcer: float = 0.03,
    n_steps: int = 200,
) -> Tuple[float, float, float]:
    """Sweep thresholds from 0 to 1 and find the one where BPCER ≈ target.

    Returns:
        best_threshold, best_apcer, best_bpcer
    """
    thresholds = np.linspace(0.01, 0.99, n_steps)
    best_threshold = 0.5
    best_apcer = 1.0
    best_bpcer = 1.0
    best_diff = float("inf")

    for t in thresholds:
        apcer, bpcer = compute_apcer_bpcer(scores, labels, t)
        diff = abs(bpcer - target_bpcer)
        if diff < best_diff:
            best_diff = diff
            best_threshold = t
            best_apcer = apcer
            best_bpcer = bpcer

    return best_threshold, best_apcer, best_bpcer


def compute_eer(
    scores: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, float]:
    """Compute Equal Error Rate (EER) and its threshold.

    EER is the point where APCER == BPCER.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    diffs = np.abs(fpr - fnr)
    eer_idx = np.argmin(diffs)
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]

    return float(eer), float(eer_threshold)


def collect_scores_and_labels(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run model on a loader and collect prediction scores and ground-truth labels."""
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_scores.extend(probs)
            all_labels.extend(labels.numpy())

    return np.array(all_scores), np.array(all_labels)


def plot_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    output_dir: str,
) -> None:
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im)
    classes = ["LIVE", "SPOOF"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Confusion matrix saved to {path}")


def plot_roc_curve(
    labels: np.ndarray,
    scores: np.ndarray,
    output_dir: str,
) -> None:
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#2196F3", lw=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (APCER)")
    plt.ylabel("True Positive Rate (1 - BPCER)")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] ROC curve saved to {path}")


def plot_apcer_bpcer_curve(
    labels: np.ndarray,
    scores: np.ndarray,
    output_dir: str,
) -> Tuple[float, float, float]:
    """Plot APCER vs BPCER across thresholds and mark the operating point."""
    thresholds = np.linspace(0.01, 0.99, 200)
    apcers = []
    bpcers = []

    for t in thresholds:
        apcer, bpcer = compute_apcer_bpcer(scores, labels, t)
        apcers.append(apcer)
        bpcers.append(bpcer)

    apcers = np.array(apcers)
    bpcers = np.array(bpcers)

    best_t, best_apcer, best_bpcer = find_threshold_at_target_bpcer(
        scores, labels, target_bpcer=0.03, n_steps=200
    )

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, apcers, label="APCER", color="#FF5722", linewidth=2)
    plt.plot(thresholds, bpcers, label="BPCER", color="#4CAF50", linewidth=2)
    plt.axvline(x=best_t, color="gray", linestyle="--", alpha=0.7,
                label=f"Threshold = {best_t:.3f}")
    plt.axhline(y=0.03, color="red", linestyle=":", alpha=0.5, label="Target BPCER = 3%")
    plt.xlabel("Threshold")
    plt.ylabel("Error Rate")
    plt.title("APCER / BPCER vs Threshold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "apcer_bpcer_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] APCER/BPCER curve saved to {path}")

    return float(best_t), float(best_apcer), float(best_bpcer)


def plot_score_distribution(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    output_dir: str,
) -> None:
    """Plot histogram of prediction scores for LIVE and SPOOF classes."""
    live_scores = scores[labels == 0]
    spoof_scores = scores[labels == 1]

    plt.figure(figsize=(9, 5))
    plt.hist(live_scores, bins=40, alpha=0.7, label="LIVE", color="#4CAF50", edgecolor="black")
    plt.hist(spoof_scores, bins=40, alpha=0.7, label="SPOOF", color="#FF5722", edgecolor="black")
    plt.axvline(x=threshold, color="black", linestyle="--", linewidth=2,
                label=f"Threshold = {threshold:.3f}")
    plt.xlabel("Predicted Probability (SPOOF)")
    plt.ylabel("Count")
    plt.title("Score Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "score_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Score distribution saved to {path}")


def generate_gradcam_examples(
    model: nn.Module,
    test_ds: torch.utils.data.Subset,
    device: torch.device,
    output_dir: str,
    n_examples: int = 6,
) -> None:
    """Generate Grad-CAM visualizations for a few test samples."""
    os.makedirs(output_dir, exist_ok=True)
    full_ds = test_ds.dataset
    indices = test_ds.indices
    chosen = np.random.choice(len(test_ds), min(n_examples, len(test_ds)), replace=False)

    for i, idx in enumerate(chosen):
        img_tensor, label = test_ds[idx]
        img_tensor_batch = img_tensor.unsqueeze(0).to(device)

        heatmap = compute_gradcam(model, img_tensor_batch, target_class=1)

        orig_idx = indices[idx]
        orig_path = full_ds.samples[orig_idx][0]
        orig_img = cv2.imread(orig_path)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        orig_img = cv2.resize(orig_img, (224, 224))

        overlay = overlay_gradcam(orig_img, heatmap, alpha=0.5)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(orig_img)
        axes[0].set_title(f"Original ({'SPOOF' if label == 1 else 'LIVE'})")
        axes[0].axis("off")

        axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        path = os.path.join(output_dir, f"gradcam_{i}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"[INFO] Grad-CAM examples saved to {output_dir}")


def main() -> None:
    cfg = Config()
    device = torch.device(cfg.device_str)
    print(f"[INFO] Using device: {device}")

    os.makedirs(cfg.output_dir, exist_ok=True)

    _, _, test_loader = create_data_loaders(cfg)
    test_ds = test_loader.dataset

    model = LivenessNet(pretrained=False, dropout=cfg.dropout).to(device)
    model.load_state_dict(torch.load(cfg.model_save_path, map_location=device))
    model.eval()

    scores, labels = collect_scores_and_labels(model, test_loader, device)

    predictions = (scores >= 0.5).astype(int)
    print("\n" + classification_report(labels, predictions, target_names=["LIVE", "SPOOF"]))

    plot_confusion_matrix(labels, predictions, cfg.output_dir)
    plot_roc_curve(labels, scores, cfg.output_dir)
    threshold, apcer, bpcer = plot_apcer_bpcer_curve(labels, scores, cfg.output_dir)
    plot_score_distribution(labels, scores, threshold, cfg.output_dir)

    generate_gradcam_examples(model, test_ds, device, cfg.output_dir)

    eer, eer_threshold = compute_eer(scores, labels)

    results = {
        "threshold_at_3pct_bpcer": threshold,
        "apcer_at_threshold": apcer,
        "bpcer_at_threshold": bpcer,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "total_test_samples": int(len(labels)),
    }
    results_path = os.path.join(cfg.output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Evaluation results saved to {results_path}")
    print(f"  Threshold @ BPCER~3%: {threshold:.4f}")
    print(f"  APCER at that threshold: {apcer:.4f}")
    print(f"  BPCER at that threshold: {bpcer:.4f}")
    print(f"  EER: {eer:.4f} (threshold={eer_threshold:.4f})")


if __name__ == "__main__":
    main()