"""
Temperature Scaling for Model Calibration

Post-hoc calibration method that learns a single temperature parameter
to scale logits before softmax, improving probability estimates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


class TemperatureScaling(nn.Module):
    """
    Temperature scaling module.

    Learns a single scalar parameter T > 0 such that:
        calibrated_probs = softmax(logits / T)

    When T > 1: softens probabilities (reduces overconfidence)
    When T < 1: sharpens probabilities (increases confidence)
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

    def get_temperature(self):
        return self.temperature.item()


def calibrate_model(model, val_loader, device, lr=0.01, max_iter=100):
    """
    Calibrate model using temperature scaling on validation set.

    Args:
        model: Trained model (in eval mode)
        val_loader: Validation data loader
        device: Torch device
        lr: Learning rate for temperature optimization
        max_iter: Maximum optimization iterations

    Returns:
        TemperatureScaling module with learned temperature
    """
    model.eval()

    # Collect logits and labels
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Collecting logits for calibration"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            all_logits.append(logits)
            all_labels.append(labels)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Optimize temperature
    temp_scaler = TemperatureScaling().to(device)
    optimizer = torch.optim.LBFGS([temp_scaler.temperature], lr=lr, max_iter=max_iter)
    criterion = nn.CrossEntropyLoss()

    def eval_loss():
        optimizer.zero_grad()
        scaled_logits = temp_scaler(all_logits)
        loss = criterion(scaled_logits, all_labels)
        loss.backward()
        return loss

    optimizer.step(eval_loss)

    temperature = temp_scaler.get_temperature()
    print(f"Learned Temperature: {temperature:.4f}")

    return temp_scaler


def apply_temperature_scaling(logits, temperature):
    """Apply temperature scaling to logits."""
    return logits / temperature


def compute_ece(logits, labels, n_bins=15):
    """
    Compute Expected Calibration Error (ECE).

    Args:
        logits: Model logits [N, C]
        labels: True labels [N]
        n_bins: Number of bins for ECE computation

    Returns:
        ECE value (lower is better)
    """
    probs = F.softmax(logits, dim=1)
    confidences, predictions = probs.max(dim=1)
    accuracies = (predictions == labels).float()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += prop_in_bin * torch.abs(avg_confidence_in_bin - accuracy_in_bin)

    return ece.item()


def compute_reliability_diagram(logits, labels, n_bins=15):
    """
    Compute data for reliability diagram.

    Returns:
        dict with bin_centers, accuracies, confidences, counts
    """
    probs = F.softmax(logits, dim=1)
    confidences, predictions = probs.max(dim=1)
    accuracies = (predictions == labels).float()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    bin_centers = []
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        count = in_bin.sum().item()

        if count > 0:
            bin_centers.append((bin_lower + bin_upper) / 2)
            bin_accuracies.append(accuracies[in_bin].mean().item())
            bin_confidences.append(confidences[in_bin].mean().item())
            bin_counts.append(count)
        else:
            bin_centers.append((bin_lower + bin_upper) / 2)
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)
            bin_counts.append(0)

    return {
        "bin_centers": bin_centers,
        "accuracies": bin_accuracies,
        "confidences": bin_confidences,
        "counts": bin_counts,
    }


class ModelWithTemperature(nn.Module):
    """
    Wrapper that applies temperature scaling to a model's logits.
    """

    def __init__(self, model, temperature=1.0):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * temperature)

    def forward(self, x):
        logits = self.model(x)
        return logits / self.temperature

    def set_temperature(self, temperature):
        self.temperature.data.fill_(temperature)

    def get_temperature(self):
        return self.temperature.item()


def calibrate_and_wrap(model, val_loader, device, lr=0.01, max_iter=100):
    """
    Calibrate model and return wrapped model with temperature scaling.

    Returns:
        ModelWithTemperature wrapper
    """
    temp_scaler = calibrate_model(model, val_loader, device, lr, max_iter)
    wrapped_model = ModelWithTemperature(model, temp_scaler.get_temperature())
    return wrapped_model