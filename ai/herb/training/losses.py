"""
Loss functions for Herb Identification

Supports:
- CrossEntropy with class weights
- Focal Loss
- ArcFace Loss (with margin)
- Outlier Exposure Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ..config import (
    USE_ARCFACE,
    ARCFACE_MARGIN,
    ARCFACE_SCALE,
    USE_FOCAL_LOSS,
    FOCAL_GAMMA,
    LABEL_SMOOTHING,
    USE_OE,
    OE_LOSS_WEIGHT,
    CLASS_WEIGHTS,
)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class ArcFaceLoss(nn.Module):
    """
    ArcFace (Additive Angular Margin) Loss.

    Reference: "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"

    The model should output cosine similarities * scale.
    This loss applies the angular margin to the target class.
    """

    def __init__(self, margin=0.5, scale=30.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.weight = weight
        self.label_smoothing = label_smoothing

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, logits, labels):
        """
        Args:
            logits: Cosine similarities * scale [B, C] (from ArcFace head)
            labels: Target labels [B]
        """
        # Logits are already scaled cosine similarities
        # We need to apply margin to target class and compute cross entropy

        # Extract target logits
        batch_size = logits.size(0)
        target_logits = logits[torch.arange(batch_size), labels]  # [B]

        # Apply margin: cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        # Since logits = scale * cos(theta), we have cos(theta) = logits/scale
        cos_theta = target_logits / self.scale
        sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta ** 2, 0, 1))

        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

        # Threshold: if theta > pi - m, use cos(theta) - m*sin(m)
        cos_theta_m = torch.where(cos_theta > self.th, cos_theta_m, cos_theta - self.mm)

        # Convert back to logits scale
        target_logits_m = cos_theta_m * self.scale

        # Create modified logits
        modified_logits = logits.clone()
        modified_logits[torch.arange(batch_size), labels] = target_logits_m

        # Cross entropy with label smoothing
        if self.label_smoothing > 0:
            # Label smoothing for cross entropy
            n_classes = logits.size(1)
            smooth_pos = 1.0 - self.label_smoothing
            smooth_neg = self.label_smoothing / (n_classes - 1)

            log_probs = F.log_softmax(modified_logits, dim=1)
            loss = -(smooth_pos * log_probs[torch.arange(batch_size), labels] +
                     smooth_neg * log_probs.sum(dim=1).sub(log_probs[torch.arange(batch_size), labels]))
            loss = loss.mean()
        else:
            loss = F.cross_entropy(modified_logits, labels, weight=self.weight)

        return loss


class OE_Loss(nn.Module):
    """
    Outlier Exposure Loss.

    Encourages uniform predictions on out-of-distribution data.
    """

    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, logits):
        """
        Args:
            logits: Model outputs on OOD data [B, C]
        """
        K = logits.size(1)
        uniform_target = torch.full_like(logits, 1.0 / K)

        log_probs = F.log_softmax(logits / self.temperature, dim=1)
        loss = F.kl_div(log_probs, uniform_target, reduction='batchmean', log_target=False)

        return loss


def calculate_class_weights(dataset, num_classes):
    """
    Calculate class weights based on class frequency.

    Args:
        dataset: Training dataset
        num_classes: Number of classes

    Returns:
        Tensor containing class weights
    """
    class_counts = torch.zeros(num_classes)

    for _, label in dataset:
        class_counts[label] += 1

    # Avoid division by zero
    class_counts[class_counts == 0] = 1

    total_samples = class_counts.sum()

    # Effective number of samples weighting
    class_weights = total_samples / (num_classes * class_counts)

    return class_weights


def build_loss(dataset, num_classes, device):
    """
    Create loss function based on configuration.

    Args:
        dataset
        num_classes
        device

    Returns:
        Loss function (or dict of losses)
    """

    # Compute class weights
    if CLASS_WEIGHTS is None:
        weights = calculate_class_weights(dataset, num_classes).to(device)
    else:
        weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(device)

    print(f"Class weights: {weights.cpu().numpy()}")

    losses = {}

    # Main classification loss
    if USE_ARCFACE:
        # ArcFace loss is applied in the model's forward pass
        # Here we just use cross entropy on the modified logits
        criterion = ArcFaceLoss(
            margin=ARCFACE_MARGIN,
            scale=ARCFACE_SCALE,
            weight=weights,
            label_smoothing=LABEL_SMOOTHING,
        )
        print(f"Using ArcFace Loss (margin={ARCFACE_MARGIN}, scale={ARCFACE_SCALE})")
    elif USE_FOCAL_LOSS:
        criterion = FocalLoss(
            gamma=FOCAL_GAMMA,
            weight=weights,
            label_smoothing=LABEL_SMOOTHING,
        )
        print(f"Using Focal Loss (gamma={FOCAL_GAMMA})")
    else:
        criterion = nn.CrossEntropyLoss(
            weight=weights,
            label_smoothing=LABEL_SMOOTHING,
        )
        print(f"Using CrossEntropy Loss (label_smoothing={LABEL_SMOOTHING})")

    losses['cls'] = criterion

    # Outlier Exposure loss
    if USE_OE:
        losses['oe'] = OE_Loss()
        print(f"Using Outlier Exposure Loss (weight={OE_LOSS_WEIGHT})")

    return losses