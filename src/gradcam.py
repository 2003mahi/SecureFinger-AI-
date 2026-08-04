import numpy as np
import cv2
import torch
import torch.nn.functional as F


def compute_gradcam(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_class: int | None = None,
) -> np.ndarray:
    """Compute Grad-CAM heatmap for a single input image.

    Uses a forward hook to capture features and a backward pre-hook
    to capture gradients, which works even when the backbone features are frozen.

    Args:
        model: Trained LivenessNet.
        image_tensor: Preprocessed tensor of shape (1, 3, 224, 224).
        target_class: Index of the class to visualise. If None, uses argmax.

    Returns:
        Heatmap of shape (224, 224) with values in [0, 1].
    """
    model.eval()

    was_frozen = not next(model.features.parameters()).requires_grad
    if was_frozen:
        for param in model.features.parameters():
            param.requires_grad = True

    features = None
    grads = None

    def save_features(module, input, output):
        nonlocal features
        features = output.detach()

    def save_grads(module, grad_output):
        nonlocal grads
        grads = grad_output[0].detach()

    last_layer = model.features[-1]
    fwd_handle = last_layer.register_forward_hook(save_features)
    bwd_handle = last_layer.register_full_backward_pre_hook(save_grads)

    output = model(image_tensor)

    if target_class is None:
        target_class = output.argmax(dim=1).item()

    model.zero_grad()
    score = output[0, target_class]
    score.backward()

    fwd_handle.remove()
    bwd_handle.remove()

    if was_frozen:
        for param in model.features.parameters():
            param.requires_grad = False

    if features is None or grads is None:
        raise RuntimeError("Failed to compute Grad-CAM.")

    weights = torch.mean(grads, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * features, dim=1).squeeze(0)

    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    cam_np = cam.cpu().numpy()
    cam_resized = cv2.resize(cam_np, (224, 224))

    return cam_resized


def overlay_gradcam(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay a Grad-CAM heatmap on the original RGB image.

    Args:
        original_image: HxWx3 uint8 RGB image.
        heatmap: HxW float32 array in [0, 1].
        alpha: Blending factor for the heatmap overlay.

    Returns:
        Overlay image as HxWx3 uint8.
    """
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_resized = cv2.resize(heatmap_colored, (original_image.shape[1], original_image.shape[0]))
    overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_resized, alpha, 0)
    return overlay