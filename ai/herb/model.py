try:
    import timm
except ImportError:
    timm = None
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .config import (
    MODEL_NAME,
    PRETRAINED,
    USE_ARCFACE,
    ARCFACE_MARGIN,
    ARCFACE_SCALE,
)


class ArcFace(nn.Module):
    """
    ArcFace (Additive Angular Margin) Loss Head

    Reference: "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
    """

    def __init__(self, in_features, out_features, margin=0.5, scale=30.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.margin = margin
        self.scale = scale

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, features, labels=None):
        """
        Args:
            features: Normalized features [B, D]
            labels: Ground truth labels [B] (required for training)

        Returns:
            logits: [B, C] if labels provided, else cosine similarity
        """
        # Normalize features and weights
        features = F.normalize(features)
        weight = F.normalize(self.weight)

        # Cosine similarity
        cosine = F.linear(features, weight)  # [B, C]

        if labels is None:
            # Inference: return scaled cosine
            return cosine * self.scale

        # Training: apply angular margin
        sine = torch.sqrt(1.0 - torch.clamp(cosine ** 2, 0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        # Make sure phi is in valid range
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Convert labels to one-hot
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # Apply margin only to target class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.scale

        return output


class HerbClassifier(nn.Module):
    """
    Herb Identification Model using EfficientNetV2-S with optional ArcFace head.
    """

    def __init__(self, num_classes: int, use_arcface: bool = USE_ARCFACE):
        super().__init__()

        # --------------------------------------------------
        # Backbone
        # --------------------------------------------------

        if timm is None:
            raise ImportError("timm is not installed.")

        self.backbone = timm.create_model(
            MODEL_NAME,
            pretrained=PRETRAINED,
            num_classes=0,  # Remove classifier, we'll add our own
            global_pool='avg',  # Global average pooling
        )

        # Get feature dimension
        self.feature_dim = self.backbone.num_features

        # --------------------------------------------------
        # Classifier Head
        # --------------------------------------------------

        self.use_arcface = use_arcface

        if use_arcface:
            # ArcFace head
            self.arcface = ArcFace(
                in_features=self.feature_dim,
                out_features=num_classes,
                margin=ARCFACE_MARGIN,
                scale=ARCFACE_SCALE,
            )
        else:
            # Standard classifier
            self.classifier = nn.Sequential(
                nn.Dropout(0.30),
                nn.Linear(self.feature_dim, num_classes)
            )

    def forward(self, x, labels=None):
        """
        Args:
            x: Input images [B, C, H, W]
            labels: Labels for ArcFace training [B] (optional)

        Returns:
            logits: [B, num_classes]
        """
        # Extract features
        features = self.backbone(x)  # [B, D]

        if self.use_arcface:
            # ArcFace requires labels during training
            if self.training and labels is not None:
                return self.arcface(features, labels)
            else:
                # Inference: return cosine similarity * scale
                return self.arcface(features)
        else:
            return self.classifier(features)

    def get_features(self, x):
        """Extract backbone features for OOD detection."""
        return self.backbone(x)


def build_model(num_classes: int, use_arcface: bool = USE_ARCFACE):
    """
    Build Herb Identification Model

    Args:
        num_classes: Number of herb classes
        use_arcface: Whether to use ArcFace head

    Returns:
        HerbClassifier model
    """
    model = HerbClassifier(num_classes, use_arcface)
    return model