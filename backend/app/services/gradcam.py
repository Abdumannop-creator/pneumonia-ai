"""
Grad-CAM heatmap generation service.
Creates visual heatmaps showing which regions of the X-ray image
the AI model focused on when making its prediction.
"""

import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """Gradient-weighted Class Activation Mapping for CNN models."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: Preprocessed image tensor (1, C, H, W)
            class_idx: Target class index. If None, uses predicted class.

        Returns:
            numpy array heatmap (H, W) normalized to [0, 1]
        """
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward()

        # Pool gradients across spatial dimensions
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)

        # Normalize
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam


def get_target_layer(model, model_name: str):
    """Get the last convolutional layer for Grad-CAM based on model architecture."""
    model_name = model_name.lower()

    if "densenet" in model_name:
        return model.features[-1]
    elif "resnet" in model_name:
        return model.layer4[-1]
    elif "efficientnet" in model_name:
        return model.features[-1]
    else:
        raise ValueError(f"Unsupported model for Grad-CAM: {model_name}")


def generate_heatmap_overlay(
    model,
    model_name: str,
    input_tensor: torch.Tensor,
    original_image_path: str,
    save_dir: str,
    class_idx: int = 1,
    alpha: float = 0.45,
) -> dict:
    """
    Generate and save Grad-CAM heatmap overlay on the original image.

    Args:
        model: Trained PyTorch model
        model_name: Model architecture name
        input_tensor: Preprocessed image tensor
        original_image_path: Path to original X-ray image
        save_dir: Directory to save heatmap
        class_idx: Target class (1 = pneumonia)
        alpha: Overlay transparency

    Returns:
        dict with heatmap_path and affected_area_percent
    """
    try:
        target_layer = get_target_layer(model, model_name)
        grad_cam = GradCAM(model, target_layer)

        # Enable gradients for inference
        input_tensor.requires_grad_(True)
        cam = grad_cam.generate(input_tensor, class_idx=class_idx)

        # Load original image
        original = cv2.imread(original_image_path)
        if original is None:
            original = np.array(Image.open(original_image_path).convert("RGB"))
            original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)

        h, w = original.shape[:2]

        # Resize CAM to match original image
        cam_resized = cv2.resize(cam, (w, h))

        # Create colored heatmap
        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam_resized), cv2.COLORMAP_JET
        )

        # Overlay
        overlay = cv2.addWeighted(original, 1 - alpha, heatmap, alpha, 0)

        # Add border and label
        overlay = cv2.copyMakeBorder(
            overlay, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(100, 100, 255)
        )

        # Calculate affected area percentage
        threshold = 0.3
        affected_pixels = np.sum(cam_resized > threshold)
        total_pixels = cam_resized.size
        affected_area_percent = round((affected_pixels / total_pixels) * 100, 1)

        # Save heatmap
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        filename = f"heatmap_{uuid.uuid4().hex[:12]}.png"
        full_path = save_path / filename
        cv2.imwrite(str(full_path), overlay)

        return {
            "heatmap_path": str(full_path),
            "heatmap_filename": filename,
            "affected_area_percent": affected_area_percent,
        }

    except Exception as e:
        print(f"Grad-CAM generation failed: {e}")
        return {
            "heatmap_path": None,
            "heatmap_filename": None,
            "affected_area_percent": 0.0,
        }


def generate_demo_heatmap(
    original_image_path: str, save_dir: str, prob_pneumonia: float
) -> dict:
    """
    Generate a simulated heatmap for demo mode (when no trained model is available).
    Creates a realistic-looking heatmap based on the pneumonia probability.
    """
    try:
        original = cv2.imread(original_image_path)
        if original is None:
            original = np.array(Image.open(original_image_path).convert("RGB"))
            original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)

        h, w = original.shape[:2]

        # Create a simulated Gaussian heatmap centered on lung area
        # (typically lower-right region for pneumonia)
        y_center = int(h * 0.55)
        x_center = int(w * 0.55)
        sigma_y = int(h * 0.25)
        sigma_x = int(w * 0.2)

        y_coords, x_coords = np.ogrid[:h, :w]
        cam = np.exp(
            -(
                ((y_coords - y_center) ** 2) / (2 * sigma_y**2)
                + ((x_coords - x_center) ** 2) / (2 * sigma_x**2)
            )
        )

        # Add second hotspot for more realistic look
        y2 = int(h * 0.45)
        x2 = int(w * 0.4)
        cam2 = np.exp(
            -(
                ((y_coords - y2) ** 2) / (2 * (sigma_y * 0.7) ** 2)
                + ((x_coords - x2) ** 2) / (2 * (sigma_x * 0.7) ** 2)
            )
        )
        cam = np.maximum(cam, cam2 * 0.6)

        # Scale by probability
        cam = cam * prob_pneumonia
        if cam.max() > 0:
            cam = cam / cam.max()

        # Create colored heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(original, 0.55, heatmap, 0.45, 0)
        overlay = cv2.copyMakeBorder(
            overlay, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(100, 100, 255)
        )

        # Affected area
        affected_area_percent = round(prob_pneumonia * 40, 1)

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        filename = f"heatmap_{uuid.uuid4().hex[:12]}.png"
        full_path = save_path / filename
        cv2.imwrite(str(full_path), overlay)

        return {
            "heatmap_path": str(full_path),
            "heatmap_filename": filename,
            "affected_area_percent": affected_area_percent,
        }

    except Exception as e:
        print(f"Demo heatmap generation failed: {e}")
        return {
            "heatmap_path": None,
            "heatmap_filename": None,
            "affected_area_percent": 0.0,
        }
