"""
Out-of-Distribution (OOD) Detection for Image Classifiers

Implements multiple OOD detection methods:
1. Energy-based scoring (log-sum-exp of logits)
2. Mahalanobis distance in feature space
3. Maximum Softmax Probability (MSP) baseline
4. Entropy-based scoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.covariance import EmpiricalCovariance
from scipy.spatial.distance import mahalanobis
from tqdm import tqdm
from typing import Optional


class OODDetector:
    """
    Base class for OOD detectors.
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

    def get_features(self, images):
        """Extract penultimate layer features."""
        raise NotImplementedError

    def compute_scores(self, images):
        """Compute OOD scores (higher = more likely OOD)."""
        raise NotImplementedError


class EnergyBasedOOD(OODDetector):
    """
    Energy-based OOD detection.

    Energy score: E(x) = -T * log(sum(exp(f(x)/T)))
    where f(x) are the logits.

    Lower energy = more likely in-distribution
    Higher energy = more likely OOD

    Reference: "Energy-based Out-of-distribution Detection" (Liu et al., 2020)
    """

    def __init__(self, model, device, temperature=1.0):
        super().__init__(model, device)
        self.temperature = temperature

    def compute_energy(self, logits):
        """Compute energy scores from logits."""
        # Energy = -T * log(sum(exp(logits/T)))
        # For numerical stability, use logsumexp
        return -self.temperature * torch.logsumexp(logits / self.temperature, dim=1)

    def compute_scores(self, images):
        """
        Compute energy scores by running the model on raw images.

        NOTE: This performs a full forward pass through ``self.model``.
        Use :meth:`compute_from_logits` instead when logits are already
        available, to avoid redundant inference.
        """
        with torch.no_grad():
            logits = self.model(images)
            energy = self.compute_energy(logits)
        return energy.cpu().numpy()

    def compute_from_logits(self, logits):
        """
        Compute energy scores from already-computed logits.

        Numerically equivalent to ``compute_scores(images)`` when the
        logits were produced by ``self.model(images)`` under ``torch.no_grad()``,
        subject only to floating-point ordering.

        Args:
            logits: Pre-computed model logits, shape ``[N, C]``.

        Returns:
            numpy.ndarray of shape ``[N]`` with energy scores.
        """
        with torch.no_grad():
            energy = self.compute_energy(logits)
        return energy.detach().cpu().numpy()


class MahalanobisOOD(OODDetector):
    """
    Mahalanobis distance-based OOD detection.

    Fits a Gaussian distribution to penultimate layer features
    for each class, then computes Mahalanobis distance to nearest class.

    Reference: "A Simple Unified Framework for Detecting Out-of-Distribution
    Samples and Adversarial Attacks" (Lee et al., 2018)
    """

    def __init__(self, model, device, feature_layer=None):
        super().__init__(model, device)
        self.feature_layer = feature_layer
        self.class_means = None
        self.precision_matrix = None
        self.fitted = False

    def extract_features(self, loader):
        """Extract features and labels from a data loader."""
        features_list = []
        labels_list = []

        # Hook to capture penultimate features
        features = {}

        def hook(module, input, output):
            features['feats'] = output

        # Register hook on the target layer
        if self.feature_layer is not None:
            handle = self.feature_layer.register_forward_hook(hook)
        else:
            # Default: try to find the last layer before classifier
            handle = None
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear) and module.out_features <= 1000:
                    # This is likely the classifier
                    parent = dict(self.model.named_modules())[name.rsplit('.', 1)[0]]
                    for child_name, child in parent.named_children():
                        if child is module:
                            prev_children = list(parent.named_children())
                            idx = [n for n, _ in prev_children].index(child_name)
                            if idx > 0:
                                target_layer = prev_children[idx - 1][1]
                                handle = target_layer.register_forward_hook(hook)
                            break

        with torch.no_grad():
            for images, labels in tqdm(loader, desc="Extracting features"):
                images = images.to(self.device, non_blocking=True)
                _ = self.model(images)

                if 'feats' in features:
                    feats = features['feats']
                    # Global average pooling if needed
                    if feats.dim() == 4:
                        feats = feats.mean(dim=[2, 3])
                    features_list.append(feats.cpu())
                    labels_list.append(labels)

        if handle is not None:
            handle.remove()

        if not features_list:
            raise ValueError("No features extracted. Check feature_layer parameter.")

        return torch.cat(features_list, dim=0), torch.cat(labels_list, dim=0)

    def fit(self, train_loader):
        """Fit Gaussian distributions to in-distribution features."""
        features, labels = self.extract_features(train_loader)

        num_classes = labels.max().item() + 1
        self.class_means = []
        covs = []

        for c in range(num_classes):
            class_features = features[labels == c]
            if len(class_features) > 0:
                mean = class_features.mean(dim=0)
                self.class_means.append(mean)
                # Compute covariance
                centered = class_features - mean
                cov = (centered.T @ centered) / (len(class_features) - 1)
                # Add regularization
                cov += torch.eye(cov.size(0)) * 1e-4
                covs.append(cov)

        # Use tied covariance (shared across classes)
        self.class_means = torch.stack(self.class_means)
        tied_cov = torch.stack(covs).mean(dim=0)
        self.precision_matrix = torch.inverse(tied_cov)

        self.fitted = True
        print(f"Fitted Mahalanobis OOD detector for {num_classes} classes")

    def compute_mahalanobis(self, features):
        """Compute Mahalanobis distance to nearest class mean."""
        if not self.fitted:
            raise ValueError("Detector not fitted. Call fit() first.")

        # features: [N, D]
        # class_means: [C, D]
        # precision: [D, D]

        diff = features.unsqueeze(1) - self.class_means.unsqueeze(0)  # [N, C, D]
        # Mahalanobis: sqrt((x-mu)^T Sigma^-1 (x-mu))
        mahal = torch.sqrt(
            (diff @ self.precision_matrix.to(features.device) * diff).sum(dim=2)
        )  # [N, C]

        # Minimum distance across classes
        min_dist, _ = mahal.min(dim=1)
        return min_dist.cpu().numpy()

    def compute_scores(self, images):
        """Compute Mahalanobis scores (higher = more OOD)."""
        # This requires feature extraction hook
        # For inference, we'd typically pre-compute features
        raise NotImplementedError("Use compute_mahalanobis with pre-extracted features")


class MSPBasedOOD(OODDetector):
    """
    Maximum Softmax Probability (MSP) baseline.

    Simple but effective: OOD samples tend to have lower max softmax probability.
    """

    def compute_scores(self, images):
        """
        Compute 1 - max_softmax by running the model on raw images.

        NOTE: This performs a full forward pass through ``self.model``.
        Use :meth:`compute_from_probs` instead when probabilities are
        already available, to avoid redundant inference.
        """
        with torch.no_grad():
            logits = self.model(images)
            probs = F.softmax(logits, dim=1)
            max_probs, _ = probs.max(dim=1)
        return (1 - max_probs).cpu().numpy()

    def compute_from_probs(self, probs):
        """
        Compute 1 - max_softmax from already-computed probabilities.

        Numerically equivalent to ``compute_scores(images)`` when the
        probabilities were produced by ``softmax(self.model(images))``
        under ``torch.no_grad()``.
        """
        with torch.no_grad():
            max_probs, _ = probs.max(dim=1)
        return (1 - max_probs).detach().cpu().numpy()


class EntropyBasedOOD(OODDetector):
    """
    Entropy-based OOD detection.

    OOD samples tend to have higher predictive entropy.
    """

    def compute_scores(self, images):
        """
        Compute predictive entropy by running the model on raw images.

        NOTE: This performs a full forward pass through ``self.model``.
        Use :meth:`compute_from_probs` instead when probabilities are
        already available, to avoid redundant inference.
        """
        with torch.no_grad():
            logits = self.model(images)
            probs = F.softmax(logits, dim=1)
            # Entropy = -sum(p * log(p))
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
        return entropy.cpu().numpy()

    def compute_from_probs(self, probs):
        """
        Compute predictive entropy from already-computed probabilities.

        Numerically equivalent to ``compute_scores(images)`` when the
        probabilities were produced by ``softmax(self.model(images))``
        under ``torch.no_grad()``.
        """
        with torch.no_grad():
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
        return entropy.detach().cpu().numpy()


class CombinedOODDetector:
    """
    Combines multiple OOD detectors for robust detection.
    """

    def __init__(self, model, device, feature_layer=None):
        self.model = model
        self.device = device

        self.energy_detector = EnergyBasedOOD(model, device)
        self.msp_detector = MSPBasedOOD(model, device)
        self.entropy_detector = EntropyBasedOOD(model, device)

        self.mahalanobis_detector = None
        if feature_layer is not None:
            self.mahalanobis_detector = MahalanobisOOD(model, device, feature_layer)

    def fit_mahalanobis(self, train_loader):
        """Fit Mahalanobis detector if available."""
        if self.mahalanobis_detector is not None:
            self.mahalanobis_detector.fit(train_loader)

    def compute_all_scores(self, images):
        """
        Compute scores from all detectors by running the model on raw images.

        NOTE: This performs three full forward passes through ``self.model``
        (one per detector). Use :meth:`compute_all_from_logits` instead when
        logits are already available, to avoid redundant inference.
        """
        scores = {
            'energy': self.energy_detector.compute_scores(images),
            'msp': self.msp_detector.compute_scores(images),
            'entropy': self.entropy_detector.compute_scores(images),
        }
        return scores

    def compute_all_from_logits(self, logits, probs):
        """
        Compute scores for energy, msp, and entropy from already-computed
        logits and probabilities. Performs NO model forward pass.

        Args:
            logits: Pre-computed model logits, shape ``[N, C]``.
            probs: Pre-computed softmax probabilities, shape ``[N, C]``.
                Should be ``F.softmax(logits, dim=1)``.

        Returns:
            dict mapping detector name to numpy.ndarray of shape ``[N]``.
        """
        return {
            'energy': self.energy_detector.compute_from_logits(logits),
            'msp': self.msp_detector.compute_from_probs(probs),
            'entropy': self.entropy_detector.compute_from_probs(probs),
        }

    def _combine(
        self,
        scores: dict,
        weights: Optional[dict] = None,
        thresholds: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Weighted combination of pre-computed OOD scores.

        Each score is normalized by its absolute threshold so the result is
        meaningful regardless of batch size. For a single image, a value of
        1.0 means the average normalized score equals the threshold.
        """
        if weights is None:
            weights = {'energy': 1.0, 'msp': 1.0, 'entropy': 1.0}

        if thresholds is None:
            thresholds = {'energy': 10.0, 'msp': 0.5, 'entropy': 1.5}

        # Determine a reference array for shape/zeros from the first score.
        first_score = next(iter(scores.values()))
        weight_sum = 0.0
        combined = np.zeros_like(first_score)

        for name, score in scores.items():
            if name in weights and name in thresholds:
                threshold = thresholds[name]
                if threshold > 0:
                    score_norm = score / threshold
                else:
                    score_norm = score
                combined = combined + weights[name] * score_norm
                weight_sum += weights[name]

        if weight_sum > 0:
            combined = combined / weight_sum

        return combined

    def combine_scores(
        self,
        energy: np.ndarray,
        msp: np.ndarray,
        entropy: np.ndarray,
        weights: Optional[dict] = None,
        thresholds: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Compute the weighted combined OOD score from already-computed
        per-detector scores. Performs NO model forward pass.

        Numerically equivalent to ``compute_combined_score(images, ...)``
        when the inputs were produced from the same ``images`` via
        :meth:`compute_all_from_logits` (or :meth:`compute_all_scores`).

        Args:
            energy: Energy scores (numpy array of shape ``[N]``).
            msp: MSP scores (numpy array of shape ``[N]``).
            entropy: Entropy scores (numpy array of shape ``[N]``).
            weights: Optional dict of per-detector weights.
            thresholds: Optional dict of per-detector thresholds.

        Returns:
            numpy.ndarray of shape ``[N]`` with combined OOD scores.
        """
        return self._combine(
            {'energy': energy, 'msp': msp, 'entropy': entropy},
            weights=weights,
            thresholds=thresholds,
        )

    def compute_combined_score(self, images, weights=None, thresholds=None):
        """
        Compute weighted combination of scores by running each detector on
        raw images.

        NOTE: This performs three full forward passes through ``self.model``
        (one per detector). Use :meth:`combine_scores` with pre-computed
        scores when logits are already available, to avoid redundant inference.
        """
        scores = self.compute_all_scores(images)
        return self._combine(scores, weights=weights, thresholds=thresholds)


def compute_ood_metrics(id_scores, ood_scores):
    """
    Compute OOD detection metrics (AUROC, AUPR, FPR@95TPR).

    Args:
        id_scores: Scores for in-distribution samples (lower = more ID)
        ood_scores: Scores for OOD samples (higher = more OOD)

    Returns:
        Dict with AUROC, AUPR, FPR@95TPR
    """
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

    # Labels: 0 for ID, 1 for OOD
    y_true = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    y_scores = np.concatenate([id_scores, ood_scores])

    # For metrics, we want higher score = more OOD
    # So we use y_scores directly

    auroc = roc_auc_score(y_true, y_scores)
    aupr = average_precision_score(y_true, y_scores)

    # FPR at 95% TPR
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    idx = np.argmax(tpr >= 0.95)
    fpr_at_95 = fpr[idx] if idx < len(fpr) else 1.0

    return {
        'auroc': auroc,
        'aupr': aupr,
        'fpr_at_95_tpr': fpr_at_95,
    }


def evaluate_ood_detector(detector, id_loader, ood_loader, device):
    """
    Evaluate OOD detector on ID and OOD data.

    Returns:
        Dict with metrics and raw scores
    """
    detector.model.eval()

    def get_scores(loader, desc):
        all_scores = []
        with torch.no_grad():
            for images, _ in tqdm(loader, desc=desc):
                images = images.to(device, non_blocking=True)
                scores = detector.compute_scores(images)
                all_scores.extend(scores)
        return np.array(all_scores)

    id_scores = get_scores(id_loader, "Evaluating ID")
    ood_scores = get_scores(ood_loader, "Evaluating OOD")

    metrics = compute_ood_metrics(id_scores, ood_scores)

    return {
        'metrics': metrics,
        'id_scores': id_scores,
        'ood_scores': ood_scores,
    }