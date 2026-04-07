"""
AI Inference Service for pneumonia detection.
Loads the trained PyTorch model, preprocesses X-ray images, 
runs predictions, and returns classification results.
Supports demo mode when no trained model is available.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # pneumonia-ai/
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pt"
META_PATH = PROJECT_ROOT / "models" / "model_meta.json"


# Default configuration (used when meta.json is not available)
DEFAULT_META = {
    "model_name": "densenet121",
    "img_size": 224,
    "classes": ["NORMAL", "PNEUMONIA"],
    "class_to_idx": {"NORMAL": 0, "PNEUMONIA": 1},
    "threshold": 0.5,
}


class PneumoniaPredictor:
    """Singleton predictor that loads model once and runs inference."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.model = None
        self.meta = None
        self.device = "cpu"
        self.demo_mode = False
        self._load()

    def _load(self):
        """Load model and metadata from disk."""
        # Load metadata
        if META_PATH.exists():
            self.meta = json.loads(META_PATH.read_text(encoding="utf-8"))
            print(f"[Inference] Model meta loaded: {self.meta['model_name']}")
        else:
            self.meta = DEFAULT_META.copy()
            print("[Inference] No model_meta.json found, using defaults")

        # Load model
        if MODEL_PATH.exists():
            try:
                from ml.models.model_factory import create_model

                self.model = create_model(
                    self.meta["model_name"], num_classes=len(self.meta["classes"])
                )
                state_dict = torch.load(MODEL_PATH, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                print(f"[Inference] Model loaded: {MODEL_PATH}")
            except Exception as e:
                print(f"[Inference] Failed to load model: {e}")
                self.demo_mode = True
        else:
            print(f"[Inference] Model file not found: {MODEL_PATH}")
            print("[Inference] Running in DEMO MODE — results are simulated")
            self.demo_mode = True

    def get_transform(self):
        """Get image preprocessing transform."""
        img_size = self.meta.get("img_size", 224)
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def predict(self, image_path: str) -> dict:
        """
        Run prediction on an X-ray image.

        Args:
            image_path: Path to the X-ray image file

        Returns:
            dict with prediction_label, confidence, prob_normal,
            prob_pneumonia, model_name, model_version, preprocess_version
        """
        if self.demo_mode:
            return self._demo_predict(image_path)

        return self._real_predict(image_path)

    def _real_predict(self, image_path: str) -> dict:
        """Run real model inference."""
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        transform = self.get_transform()
        input_tensor = transform(image).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1).squeeze()

        prob_normal = float(probs[0])
        prob_pneumonia = float(probs[1])
        threshold = self.meta.get("threshold", 0.5)

        if prob_pneumonia >= threshold:
            prediction_label = "PNEUMONIA"
            confidence = prob_pneumonia
        else:
            prediction_label = "NORMAL"
            confidence = prob_normal

        return {
            "prediction_label": prediction_label,
            "confidence": round(float(confidence), 4),
            "prob_normal": round(prob_normal, 4),
            "prob_pneumonia": round(prob_pneumonia, 4),
            "model_name": self.meta["model_name"],
            "model_version": "v1.0",
            "preprocess_version": "v1.0",
            "input_tensor": input_tensor,  # for Grad-CAM
        }

    def _demo_predict(self, image_path: str) -> dict:
        """
        Simulate prediction for demo mode.
        Generates realistic-looking random results.
        """
        # Generate deterministic-ish results based on file size
        try:
            file_size = Path(image_path).stat().st_size
            seed = file_size % 10000
            rng = random.Random(seed)
        except Exception:
            rng = random.Random()

        # Simulate with slight bias toward pneumonia (more interesting demo)
        prob_pneumonia = round(rng.uniform(0.35, 0.95), 4)
        prob_normal = round(1.0 - prob_pneumonia, 4)

        if prob_pneumonia >= 0.5:
            prediction_label = "PNEUMONIA"
            confidence = prob_pneumonia
        else:
            prediction_label = "NORMAL"
            confidence = prob_normal

        # Create a dummy tensor for demo heatmap
        image = Image.open(image_path).convert("RGB")
        transform = self.get_transform()
        input_tensor = transform(image).unsqueeze(0)

        return {
            "prediction_label": prediction_label,
            "confidence": round(float(confidence), 4),
            "prob_normal": prob_normal,
            "prob_pneumonia": prob_pneumonia,
            "model_name": self.meta["model_name"] + " (demo)",
            "model_version": "demo-v1.0",
            "preprocess_version": "v1.0",
            "input_tensor": input_tensor,
        }


# Global predictor instance
_predictor = None


def get_predictor() -> PneumoniaPredictor:
    """Get or create the global predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = PneumoniaPredictor()
    return _predictor
