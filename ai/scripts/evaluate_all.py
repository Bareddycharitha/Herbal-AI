"""
Unified Evaluation Script for All Herbal-AI Models

Evaluates:
1. Universal Classifier (3-class) with OOD metrics
2. Skin Disease Classifier (22-class) with two-stage metrics
3. Herb Identification with leaf detector metrics
4. Calibration evaluation (ECE, reliability diagrams)
5. Grad-CAM quality assessment
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score,
    average_precision_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

from ai.config import DEVICE, CHECKPOINT_DIR, BEST_MODEL_PATH, RESULTS_DIR
from ai.image_classifier.config import DEVICE as UNIV_DEVICE, BEST_MODEL_PATH as UNIV_BEST
from ai.herb.config import DEVICE as HERB_DEVICE, BEST_MODEL_PATH as HERB_BEST, CHECKPOINT_DIR as HERB_CKPT

from ai.image_classifier.inference import UniversalClassifierInference
from ai.training.inference import SkinDiseaseInference
from ai.herb.inference import HerbInference
from ai.training.calibration import compute_ece, compute_reliability_diagram
from ai.training.ood_detection import evaluate_ood_detector, EnergyBasedOOD, MSPBasedOOD, EntropyBasedOOD, CombinedOODDetector


# ==========================================================
# Universal Classifier Evaluation
# ==========================================================

def evaluate_universal_classifier(
    model_path: str = None,
    ensemble_paths: List[str] = None,
    cal_path: str = None,
    test_dirs: List[str] = None,
    ood_dirs: List[str] = None,
) -> Dict:
    """
    Evaluate Universal Classifier (Skin/Medicinal/Other).

    Args:
        model_path: Single model checkpoint
        ensemble_paths: List of model checkpoints for ensemble
        cal_path: Calibration temperature path
        test_dirs: Test directories for each class [skin, medicinal, other]
        ood_dirs: OOD test directories (non-skin/non-herb images)

    Returns:
        Dict with evaluation metrics
    """
    print("\n" + "="*60)
    print("EVALUATING UNIVERSAL CLASSIFIER")
    print("="*60)

    # Load model
    classifier = UniversalClassifierInference(
        model_path=model_path,
        ensemble_paths=ensemble_paths,
        calibration_path=cal_path,
    )

    # If test_dirs not provided, use default
    if test_dirs is None:
        from ai.image_classifier.config import TEST_DIRS
        test_dirs = TEST_DIRS

    # Evaluate on test set
    from ai.image_classifier.dataset import UniversalImageDataset
    from ai.image_classifier.transforms import test_transform
    from torch.utils.data import DataLoader

    test_dataset = UniversalImageDataset(
        dataset_dirs=test_dirs,
        transform=test_transform,
    )

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Collect predictions
    all_preds = []
    all_labels = []
    all_probs = []
    all_energy = []
    all_msp = []
    all_entropy = []

    classifier.model.eval() if not classifier.use_ensemble else [m.eval() for m in classifier.models]

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Universal Test"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            if classifier.use_ensemble:
                logits = classifier._get_logits_ensemble(images)
            else:
                logits = classifier._get_logits_single(classifier.model, images)

            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu())

            # OOD scores
            energy = classifier.energy_ood.compute_scores(images)
            msp = classifier.msp_ood.compute_scores(images)
            entropy = classifier.entropy_ood.compute_scores(images)

            all_energy.extend(energy)
            all_msp.extend(msp)
            all_entropy.extend(entropy)

    all_probs = torch.cat(all_probs, dim=0)

    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    macro_precision = precision_score(all_labels, all_preds, average='macro')
    macro_recall = recall_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)

    # Per-class metrics
    class_names = ["Skin", "Medicinal", "Other"]
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    # ECE
    logits = torch.log(all_probs + 1e-10)
    ece = compute_ece(logits, torch.tensor(all_labels))

    results = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "per_class_accuracy": {name: float(acc) for name, acc in zip(class_names, per_class_acc)},
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(all_labels, all_preds, target_names=class_names, output_dict=True),
        "ece": ece,
    }

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"ECE: {ece:.4f}")
    print(f"Per-class Acc: {results['per_class_accuracy']}")

    # OOD Evaluation (if OOD dirs provided)
    if ood_dirs:
        print("\nEvaluating OOD detection...")
        # Create OOD dataset
        ood_dataset = UniversalImageDataset(
            dataset_dirs=ood_dirs,
            transform=test_transform,
        )
        ood_loader = DataLoader(ood_dataset, batch_size=32, shuffle=False, num_workers=2)

        # Evaluate energy-based OOD
        ood_results = evaluate_ood_detector(
            classifier.energy_ood,
            test_loader,
            ood_loader,
            DEVICE,
        )

        results["ood_metrics"] = ood_results["metrics"]
        print(f"OOD AUROC: {ood_results['metrics']['auroc']:.4f}")
        print(f"OOD AUPR: {ood_results['metrics']['aupr']:.4f}")
        print(f"OOD FPR@95TPR: {ood_results['metrics']['fpr_at_95_tpr']:.4f}")

    return results


# ==========================================================
# Skin Disease Classifier Evaluation
# ==========================================================

def evaluate_skin_disease(
    binary_model_path: str = None,
    multiclass_model_path: str = None,
    ensemble_paths: List[str] = None,
    cal_path: str = None,
    test_dir: str = None,
    ood_dirs: List[str] = None,
) -> Dict:
    """
    Evaluate Skin Disease Classifier (22-class + Healthy).

    Args:
        binary_model_path: Binary (Healthy/Diseased) model
        multiclass_model_path: 22-class disease model
        ensemble_paths: Ensemble model paths
        cal_path: Calibration path
        test_dir: Test directory
        ood_dirs: OOD test directories

    Returns:
        Dict with evaluation metrics
    """
    print("\n" + "="*60)
    print("EVALUATING SKIN DISEASE CLASSIFIER")
    print("="*60)

    # Load inference
    inference = SkinDiseaseInference(
        binary_model_path=binary_model_path,
        multiclass_model_path=multiclass_model_path,
        ensemble_paths=ensemble_paths,
    )

    # Load test data
    from ai.preprocessing.dataset import SkinDiseaseDataset
    from ai.preprocessing.transforms import get_valid_transforms
    from torch.utils.data import DataLoader

    if test_dir is None:
        from ai.config import TEST_DIR
        test_dir = TEST_DIR

    test_dataset = SkinDiseaseDataset(
        root_dir=test_dir,
        transform=get_valid_transforms(),
    )

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    class_names = test_dataset.classes

    # Collect predictions
    all_preds = []
    all_labels = []
    all_probs = []
    all_confidences = []
    all_energy = []
    all_msp = []
    all_entropy = []
    binary_correct = 0
    binary_total = 0

    base_model = inference.multiclass_models[0] if inference.use_ensemble else inference.multiclass_model

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Skin Disease Test"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Binary stage
            if inference.use_two_stage and inference.binary_model is not None:
                binary_logits = inference._get_binary_logits(images)
                binary_probs = F.softmax(binary_logits, dim=1)
                binary_preds = binary_probs.argmax(dim=1)

                # Label 0 = Healthy (Unknown_Normal), 1 = Diseased
                healthy_idx = test_dataset.class_to_idx.get("Unknown_Normal", -1)
                is_healthy_gt = (labels == healthy_idx)
                is_healthy_pred = (binary_preds == 0)

                binary_correct += (is_healthy_gt == is_healthy_pred).sum().item()
                binary_total += len(labels)

            # Multiclass
            logits = inference._get_multiclass_logits(images)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            confidences, _ = probs.max(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu())
            all_confidences.extend(confidences.cpu().numpy())

            # OOD scores
            energy = inference.energy_ood.compute_scores(images)
            msp = inference.msp_ood.compute_scores(images)
            entropy = inference.entropy_ood.compute_scores(images)

            all_energy.extend(energy)
            all_msp.extend(msp)
            all_entropy.extend(entropy)

    all_probs = torch.cat(all_probs, dim=0)

    # Overall metrics
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    macro_precision = precision_score(all_labels, all_preds, average='macro')
    macro_recall = recall_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)

    # Per-class
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    # ECE
    logits = torch.log(all_probs + 1e-10)
    ece = compute_ece(logits, torch.tensor(all_labels))

    # Confidence analysis
    confidences = np.array(all_confidences)
    high_conf = confidences >= 0.7
    med_conf = (confidences >= 0.4) & (confidences < 0.7)
    low_conf = confidences < 0.4

    # Healthy skin specific metrics
    healthy_idx = test_dataset.class_to_idx.get("Unknown_Normal", -1)
    healthy_mask = (np.array(all_labels) == healthy_idx)
    healthy_acc = accuracy_score(
        np.array(all_labels)[healthy_mask],
        np.array(all_preds)[healthy_mask]
    ) if healthy_mask.any() else 0

    diseased_mask = ~healthy_mask
    diseased_acc = accuracy_score(
        np.array(all_labels)[diseased_mask],
        np.array(all_preds)[diseased_mask]
    ) if diseased_mask.any() else 0

    results = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "healthy_accuracy": healthy_acc,
        "diseased_accuracy": diseased_acc,
        "per_class_accuracy": {name: float(acc) for name, acc in zip(class_names, per_class_acc)},
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(all_labels, all_preds, target_names=class_names, output_dict=True),
        "ece": ece,
        "confidence_distribution": {
            "high": float(high_conf.mean()),
            "medium": float(med_conf.mean()),
            "low": float(low_conf.mean()),
        },
        "binary_stage_accuracy": binary_correct / binary_total if binary_total > 0 else None,
    }

    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Healthy Skin Acc: {healthy_acc:.4f}")
    print(f"Diseased Acc: {diseased_acc:.4f}")
    print(f"ECE: {ece:.4f}")
    print(f"Binary Stage Acc: {results['binary_stage_accuracy']:.4f}")

    # OOD Evaluation
    if ood_dirs:
        print("\nEvaluating OOD detection...")
        # Create OOD dataset
        ood_dataset = SkinDiseaseDataset(
            root_dir=ood_dirs[0] if isinstance(ood_dirs, list) else ood_dirs,
            transform=get_valid_transforms(),
        )
        ood_loader = DataLoader(ood_dataset, batch_size=32, shuffle=False, num_workers=2)

        ood_results = evaluate_ood_detector(
            inference.energy_ood,
            test_loader,
            ood_loader,
            DEVICE,
        )

        results["ood_metrics"] = ood_results["metrics"]
        print(f"OOD AUROC: {ood_results['metrics']['auroc']:.4f}")
        print(f"OOD AUPR: {ood_results['metrics']['aupr']:.4f}")
        print(f"OOD FPR@95TPR: {ood_results['metrics']['fpr_at_95_tpr']:.4f}")

    return results


# ==========================================================
# Herb Identification Evaluation
# ==========================================================

def evaluate_herb_identification(
    model_path: str = None,
    ensemble_paths: List[str] = None,
    cal_path: str = None,
    test_dir: str = None,
    ood_dirs: List[str] = None,
    non_leaf_dirs: List[str] = None,
) -> Dict:
    """
    Evaluate Herb Identification with leaf detector.

    Args:
        model_path: Model checkpoint
        ensemble_paths: Ensemble paths
        cal_path: Calibration path
        test_dir: Test directory for herbs
        ood_dirs: OOD directories (non-herb plants)
        non_leaf_dirs: Non-leaf directories for leaf detector eval

    Returns:
        Dict with evaluation metrics
    """
    print("\n" + "="*60)
    print("EVALUATING HERB IDENTIFICATION")
    print("="*60)

    # Load inference
    inference = HerbInference(
        model_path=model_path,
        ensemble_paths=ensemble_paths,
        calibration_path=cal_path,
    )

    # Load test data
    from ai.herb.dataset import HerbDataset
    from ai.herb.transforms import test_transforms
    from torch.utils.data import DataLoader

    if test_dir is None:
        from ai.herb.config import DATASET_DIR
        test_dir = DATASET_DIR / "test"

    test_dataset = HerbDataset(
        root_dir=test_dir,
        transform=test_transforms,
    )

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    class_names = test_dataset.classes

    # Collect predictions
    all_preds = []
    all_labels = []
    all_probs = []
    all_confidences = []
    all_energy = []
    all_msp = []
    all_entropy = []
    leaf_correct = 0
    leaf_total = 0

    base_model = inference.models[0] if inference.use_ensemble else inference.model

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Herb Test"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Leaf detector
            if inference.use_leaf_detector and inference.leaf_detector is not None:
                # We can't easily batch this, so skip for batch eval
                pass

            logits = inference._get_logits_ensemble(images) if inference.use_ensemble else inference._get_logits_single(inference.model, images)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            confidences, _ = probs.max(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu())
            all_confidences.extend(confidences.cpu().numpy())

            # OOD scores
            energy = inference.energy_ood.compute_scores(images)
            msp = inference.msp_ood.compute_scores(images)
            entropy = inference.entropy_ood.compute_scores(images)

            all_energy.extend(energy)
            all_msp.extend(msp)
            all_entropy.extend(entropy)

    all_probs = torch.cat(all_probs, dim=0)

    # Overall metrics
    accuracy = accuracy_score(all_labels, all_preds)
    top3_acc = top_k_accuracy(all_probs, torch.tensor(all_labels), k=3)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    macro_precision = precision_score(all_labels, all_preds, average='macro')
    macro_recall = recall_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)

    # Per-class
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    # ECE
    logits = torch.log(all_probs + 1e-10)
    ece = compute_ece(logits, torch.tensor(all_labels))

    results = {
        "accuracy": accuracy,
        "top3_accuracy": top3_acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "per_class_accuracy": {name: float(acc) for name, acc in zip(class_names, per_class_acc)},
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(all_labels, all_preds, target_names=class_names, output_dict=True),
        "ece": ece,
    }

    print(f"Top-1 Accuracy: {accuracy:.4f}")
    print(f"Top-3 Accuracy: {top3_acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"ECE: {ece:.4f}")

    # Non-leaf evaluation (if provided)
    if non_leaf_dirs and inference.use_leaf_detector:
        print("\nEvaluating leaf detector on non-leaf images...")
        leaf_detector_results = evaluate_leaf_detector(
            inference.leaf_detector,
            non_leaf_dirs,
        )
        results["leaf_detector"] = leaf_detector_results

    return results


def top_k_accuracy(probs, labels, k=3):
    """Compute top-k accuracy."""
    _, top_k_preds = probs.topk(k, dim=1)
    correct = (top_k_preds == labels.unsqueeze(1)).any(dim=1).float()
    return correct.mean().item()


def evaluate_leaf_detector(leaf_detector, non_leaf_dirs: List[str]) -> Dict:
    """Evaluate leaf detector on non-leaf images."""
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    import torchvision.transforms as T

    class NonLeafDataset(Dataset):
        def __init__(self, dirs, transform):
            self.samples = []
            for d in dirs:
                d = Path(d)
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                    self.samples.extend(list(d.glob(f'*{ext}')))
            self.transform = transform

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path = self.samples[idx]
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, 0  # 0 = non-leaf

    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = NonLeafDataset(non_leaf_dirs, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

    correct = 0
    total = 0

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(DEVICE)
            # Leaf detector predict
            logits = leaf_detector.model(images)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            # Class 1 = leaf, Class 0 = non-leaf
            correct += (preds == 0).sum().item()
            total += len(preds)

    accuracy = correct / total if total > 0 else 0
    print(f"Leaf Detector Non-Leaf Rejection Rate: {accuracy:.4f}")

    return {"non_leaf_rejection_rate": accuracy, "total_samples": total}


# ==========================================================
# Calibration Evaluation
# ==========================================================

def evaluate_calibration(model, val_loader, device, model_name: str) -> Dict:
    """
    Evaluate model calibration.

    Returns:
        Dict with ECE, reliability diagram data, confidence histograms
    """
    print(f"\nEvaluating calibration for {model_name}...")

    model.eval()
    all_logits = []
    all_labels = []
    all_probs = []
    all_confidences = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Calibration {model_name}"):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probs = F.softmax(logits, dim=1)
            confidences, preds = probs.max(dim=1)

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())
            all_confidences.append(confidences.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_probs = torch.cat(all_probs, dim=0)
    all_confidences = torch.cat(all_confidences, dim=0)

    # ECE
    ece = compute_ece(all_logits, all_labels)

    # Reliability diagram
    rel_diagram = compute_reliability_diagram(all_logits, all_labels)

    # Confidence histograms (correct vs incorrect)
    correct = (all_probs.argmax(dim=1) == all_labels)
    conf_correct = all_confidences[correct].numpy()
    conf_incorrect = all_confidences[~correct].numpy()

    results = {
        "ece": ece,
        "reliability_diagram": rel_diagram,
        "confidence_stats": {
            "correct_mean": float(conf_correct.mean()) if len(conf_correct) > 0 else 0,
            "correct_std": float(conf_correct.std()) if len(conf_correct) > 0 else 0,
            "incorrect_mean": float(conf_incorrect.mean()) if len(conf_incorrect) > 0 else 0,
            "incorrect_std": float(conf_incorrect.std()) if len(conf_incorrect) > 0 else 0,
        },
    }

    print(f"  ECE: {ece:.4f}")
    print(f"  Correct confidence: {results['confidence_stats']['correct_mean']:.4f} ± {results['confidence_stats']['correct_std']:.4f}")
    print(f"  Incorrect confidence: {results['confidence_stats']['incorrect_mean']:.4f} ± {results['confidence_stats']['incorrect_std']:.4f}")

    return results


# ==========================================================
# Grad-CAM Evaluation
# ==========================================================

def evaluate_gradcam(model, test_loader, device, class_names, num_samples=50):
    """
    Evaluate Grad-CAM quality.

    Returns:
        Dict with localization metrics
    """
    from ai.explainability.gradcam_engine import GradCAM, overlay_heatmap
    import cv2

    print(f"\nEvaluating Grad-CAM for {len(class_names)} classes...")

    # Find target layer
    target_layer = None
    for name, module in model.named_modules():
        if 'conv_head' in name or 'features' in name or 'blocks' in name:
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module

    if target_layer is None:
        print("  Could not find target layer for Grad-CAM")
        return {}

    gradcam = GradCAM(model, target_layer)

    # Collect samples per class
    class_samples = {i: [] for i in range(len(class_names))}

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            for i in range(len(images)):
                label = labels[i].item()
                if len(class_samples[label]) < 3:  # 3 samples per class
                    class_samples[label].append(images[i].unsqueeze(0))

    # Generate Grad-CAM for collected samples
    total_generated = 0
    for class_idx, samples in class_samples.items():
        for sample in samples:
            if total_generated >= num_samples:
                break

            logits = model(sample)
            pred = logits.argmax(dim=1).item()

            try:
                cam = gradcam.generate(sample, pred)
                total_generated += 1
            except Exception as e:
                print(f"  Grad-CAM failed for class {class_idx}: {e}")

    results = {
        "samples_generated": total_generated,
        "target_layer": str(target_layer),
    }

    print(f"  Generated {total_generated} Grad-CAM visualizations")
    return results


# ==========================================================
# Visualization Functions
# ==========================================================

def plot_confusion_matrix(cm, class_names, title, save_path=None):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved confusion matrix to {save_path}")

    plt.close()


def plot_reliability_diagram(rel_diagram, title, save_path=None):
    """Plot reliability diagram."""
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.plot(rel_diagram['bin_centers'], rel_diagram['accuracies'], 'o-', label='Model')
    plt.fill_between(rel_diagram['bin_centers'],
                     rel_diagram['accuracies'],
                     rel_diagram['bin_centers'],
                     alpha=0.3)
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved reliability diagram to {save_path}")

    plt.close()


def plot_confidence_histogram(conf_correct, conf_incorrect, title, save_path=None):
    """Plot confidence histograms for correct vs incorrect predictions."""
    plt.figure(figsize=(10, 6))
    plt.hist(conf_correct, bins=20, alpha=0.5, label='Correct', density=True, color='green')
    plt.hist(conf_incorrect, bins=20, alpha=0.5, label='Incorrect', density=True, color='red')
    plt.xlabel('Confidence')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved confidence histogram to {save_path}")

    plt.close()


# ==========================================================
# Main Evaluation
# ==========================================================

def run_full_evaluation(
    output_dir: str = "evaluation_results",
    evaluate_universal: bool = True,
    evaluate_skin: bool = True,
    evaluate_herb: bool = True,
):
    """
    Run full evaluation of all models.

    Args:
        output_dir: Directory to save results
        evaluate_universal: Whether to evaluate universal classifier
        evaluate_skin: Whether to evaluate skin disease classifier
        evaluate_herb: Whether to evaluate herb identification
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # 1. Universal Classifier
    if evaluate_universal:
        print("\n" + "#"*60)
        print("# UNIVERSAL CLASSIFIER EVALUATION")
        print("#"*60)

        try:
            results = evaluate_universal_classifier()
            all_results["universal_classifier"] = results

            # Save confusion matrix
            cm = np.array(results["confusion_matrix"])
            plot_confusion_matrix(
                cm, ["Skin", "Medicinal", "Other"],
                "Universal Classifier Confusion Matrix",
                output_path / "universal_confusion_matrix.png"
            )

            # Save reliability diagram
            if "reliability_diagram" in results:
                plot_reliability_diagram(
                    results["reliability_diagram"],
                    "Universal Classifier Reliability Diagram",
                    output_path / "universal_reliability.png"
                )

        except Exception as e:
            print(f"Error evaluating universal classifier: {e}")
            all_results["universal_classifier"] = {"error": str(e)}

    # 2. Skin Disease
    if evaluate_skin:
        print("\n" + "#"*60)
        print("# SKIN DISEASE CLASSIFIER EVALUATION")
        print("#"*60)

        try:
            results = evaluate_skin_disease()
            all_results["skin_disease"] = results

            # Save confusion matrix
            cm = np.array(results["confusion_matrix"])
            class_names = list(results["per_class_accuracy"].keys())
            plot_confusion_matrix(
                cm, class_names,
                "Skin Disease Confusion Matrix",
                output_path / "skin_confusion_matrix.png"
            )

        except Exception as e:
            print(f"Error evaluating skin disease: {e}")
            all_results["skin_disease"] = {"error": str(e)}

    # 3. Herb Identification
    if evaluate_herb:
        print("\n" + "#"*60)
        print("# HERB IDENTIFICATION EVALUATION")
        print("#"*60)

        try:
            results = evaluate_herb_identification()
            all_results["herb_identification"] = results

            # Save confusion matrix (sample - too many classes for full)
            cm = np.array(results["confusion_matrix"])
            class_names = list(results["per_class_accuracy"].keys())
            plot_confusion_matrix(
                cm, class_names,
                "Herb Identification Confusion Matrix",
                output_path / "herb_confusion_matrix.png"
            )

        except Exception as e:
            print(f"Error evaluating herb identification: {e}")
            all_results["herb_identification"] = {"error": str(e)}

    # Save all results
    results_file = output_path / "evaluation_results.json"
    with open(results_file, "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.ndarray, np.generic)):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        json.dump(convert(all_results), f, indent=2)

    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to {output_path}")
    print(f"Summary:")

    for model_name, results in all_results.items():
        if "error" not in results:
            print(f"  {model_name}: Acc={results.get('accuracy', 'N/A'):.4f}, F1={results.get('macro_f1', 'N/A'):.4f}, ECE={results.get('ece', 'N/A'):.4f}")
        else:
            print(f"  {model_name}: ERROR - {results['error']}")

    return all_results


# ==========================================================
# CLI
# ==========================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Herbal-AI models")
    parser.add_argument("--output", default="evaluation_results", help="Output directory")
    parser.add_argument("--skip-universal", action="store_true", help="Skip universal classifier")
    parser.add_argument("--skip-skin", action="store_true", help="Skip skin disease")
    parser.add_argument("--skip-herb", action="store_true", help="Skip herb identification")

    args = parser.parse_args()

    run_full_evaluation(
        output_dir=args.output,
        evaluate_universal=not args.skip_universal,
        evaluate_skin=not args.skip_skin,
        evaluate_herb=not args.skip_herb,
    )