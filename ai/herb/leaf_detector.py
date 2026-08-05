"""
Binary Leaf Detector: Leaf vs Non-Leaf Classification

Lightweight MobileNetV3-Small model for filtering non-leaf images
before herb identification.
"""

import timm
import torch
import torch.nn as nn
from pathlib import Path

from .config import (
    DEVICE,
    LEAF_DETECTOR_MODEL,
    LEAF_DETECTOR_PATH,
    LEAF_DETECTOR_THRESHOLD,
    IMAGE_SIZE,
)


class LeafDetector(nn.Module):
    """
    Binary Leaf Detector using MobileNetV3-Small.
    """

    def __init__(self):
        super().__init__()

        # MobileNetV3-Small - lightweight and fast
        self.backbone = timm.create_model(
            LEAF_DETECTOR_MODEL,
            pretrained=True,
            num_classes=0,
            global_pool='avg',
        )

        # MobileNetV3-Small has conv_head that expands 576 -> 1024
        # The actual output feature dimension is 1024
        self.feature_dim = 1024  # After conv_head

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.feature_dim, 2)  # Leaf (1) vs Non-Leaf (0)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


def build_leaf_detector():
    """Build leaf detector model."""
    return LeafDetector()


class LeafDetectorInference:
    """
    Leaf Detector Inference Wrapper.
    """

    def __init__(self, model_path=None, threshold=None):
        self.device = DEVICE
        self.threshold = threshold or LEAF_DETECTOR_THRESHOLD

        if model_path is None:
            model_path = LEAF_DETECTOR_PATH

        self.model = build_leaf_detector().to(self.device)

        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Loaded leaf detector from {model_path}")
        else:
            # Checkpoint not found - model uses random weights
            # Caller should handle this case (disable leaf detector)
            pass

        self.model.eval()

        # Transforms
        from ai.herb.transforms import test_transforms
        self.transform = test_transforms

    def predict(self, image_path):
        """
        Predict if image contains a leaf.

        Returns:
            dict with:
                - is_leaf: bool
                - confidence: float (0-1)
                - leaf_prob: float (0-1)
        """
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            leaf_prob = probs[0, 1].item()  # Class 1 = Leaf

        is_leaf = leaf_prob >= self.threshold

        return {
            "is_leaf": is_leaf,
            "confidence": max(leaf_prob, 1 - leaf_prob),
            "leaf_prob": leaf_prob,
        }

    def predict_batch(self, image_paths):
        """Predict batch of images."""
        return [self.predict(p) for p in image_paths]


# Singleton instance
_leaf_detector = None


def get_leaf_detector():
    """Get or create singleton leaf detector."""
    global _leaf_detector
    if _leaf_detector is None:
        _leaf_detector = LeafDetectorInference()
    return _leaf_detector


def is_leaf_image(image_path):
    """Convenience function to check if image is a leaf."""
    detector = get_leaf_detector()
    return detector.predict(image_path)["is_leaf"]