"""
Evaluation script for Universal Image Classifier (quick version).
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from .config import TEST_DIRS, RESULTS_DIR, DEVICE
from .model import build_model
from .transforms import test_transform
from .inference import UniversalClassifierInference
from PIL import Image
import tqdm


def load_test_data() -> List[Tuple[Path, int]]:
    """
    Load all test images with their corresponding labels.

    Returns:
        List of tuples (image_path, label)
    """
    # Class mapping from dataset.py
    label_mapping = {
        "skin": 0,
        "medicinal": 1,
        "other": 2,
    }

    # Reverse mapping for dataset names
    dataset_to_label = {
        "skindisease": label_mapping["skin"],
        "medicinal_leaf": label_mapping["medicinal"],
        "medicinal_plant": label_mapping["medicinal"],
        "otherobjects": label_mapping["other"],
    }

    samples = []

    for test_dir in TEST_DIRS:
        # Determine label based on parent directory name
        dataset_name = test_dir.parent.name.lower()

        # Special handling for Medicinal datasets (both leaf and plant map to medicinal)
        if "medicinal" in dataset_name:
            label = label_mapping["medicinal"]
        elif "skin" in dataset_name:
            label = label_mapping["skin"]
        elif "other" in dataset_name:
            label = label_mapping["other"]
        else:
            # Try to map using dataset_to_label
            label = dataset_to_label.get(dataset_name, -1)
            if label == -1:
                print(f"Warning: Unknown dataset {dataset_name}, skipping")
                continue

        # Collect all image files in the test directory
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        for image_path in test_dir.rglob('*'):
            if image_path.is_file() and image_path.suffix.lower() in image_extensions:
                samples.append((image_path, label))

    # Return only first 10 samples for quick evaluation
    return samples[:10]


def evaluate_model() -> Dict:
    """
    Evaluate the universal image classifier on test data.

    Returns:
        Dictionary containing evaluation results
    """
    print("Loading test data...")
    samples = load_test_data()
    print(f"Found {len(samples)} test images (subset for quick evaluation)")

    if len(samples) == 0:
        raise ValueError("No test images found. Check TEST_DIRS in config.py")

    # Initialize classifier
    print("Loading model...")
    classifier = UniversalClassifierInference()

    # Initialize lists for storing results
    true_labels = []
    pred_labels = []
    confidences = []
    failed_predictions = []  # List of dicts for failed predictions

    # Evaluate each image
    print("Running evaluation...")
    for image_path, true_label in tqdm.tqdm(samples, desc="Evaluating"):
        try:
            # Get prediction
            result = classifier.predict(str(image_path), return_details=False)
            pred_label_str = result['class']
            confidence = result['confidence']  # Already in 0-100 scale

            # Map class name to label
            label_map = {"Skin": 0, "Medicinal": 1, "Other": 2}
            pred_label = label_map[pred_label_str]

            # Store results
            true_labels.append(true_label)
            pred_labels.append(pred_label)
            confidences.append(confidence)

            # Check if prediction is incorrect
            if true_label != pred_label:
                # Determine error type
                error_types = {
                    (0, 2): "Skin->Other",      # Skin incorrectly as Other
                    (1, 2): "Medicinal->Other", # Medicinal incorrectly as Other
                    (2, 0): "Other->Skin",      # Other incorrectly as Skin
                    (2, 1): "Other->Medicinal", # Other incorrectly as Medicinal
                    (0, 1): "Skin->Medicinal",  # Skin incorrectly as Medicinal
                    (1, 0): "Medicinal->Skin",  # Medicinal incorrectly as Skin
                }
                error_key = (true_label, pred_label)
                error_type = error_types.get(error_key, f"{true_label}->{pred_label}")

                failed_predictions.append({
                    'image_path': str(image_path),
                    'actual_class': ['Skin', 'Medicinal', 'Other'][true_label],
                    'predicted_class': ['Skin', 'Medicinal', 'Other'][pred_label],
                    'confidence': confidence,
                    'error_type': error_type
                })

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            # Still count as failed prediction with error
            true_labels.append(true_label)
            pred_labels.append(-1)  # Invalid prediction
            confidences.append(0.0)
            failed_predictions.append({
                'image_path': str(image_path),
                'actual_class': ['Skin', 'Medicinal', 'Other'][true_label],
                'predicted_class': 'Error',
                'confidence': 0.0,
                'error_type': 'ProcessingError'
            })

    # Convert to numpy arrays
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    confidences = np.array(confidences)

    # Filter out any invalid predictions (where pred_label == -1 due to error)
    valid_mask = pred_labels != -1
    if not np.all(valid_mask):
        print(f"Warning: {np.sum(~valid_mask)} images had processing errors and were excluded from metrics")
        true_labels_valid = true_labels[valid_mask]
        pred_labels_valid = pred_labels[valid_mask]
        confidences_valid = confidences[valid_mask]
        failed_predictions_valid = [fp for i, fp in enumerate(failed_predictions) if valid_mask[i]]
    else:
        true_labels_valid = true_labels
        pred_labels_valid = pred_labels
        confidences_valid = confidences
        failed_predictions_valid = failed_predictions

    # Calculate metrics
    overall_accuracy = accuracy_score(true_labels_valid, pred_labels_valid)

    # Per-class accuracy
    class_accuracies = {}
    for class_idx, class_name in enumerate(['Skin', 'Medicinal', 'Other']):
        class_mask = true_labels_valid == class_idx
        if np.sum(class_mask) > 0:
            class_acc = accuracy_score(
                true_labels_valid[class_mask],
                pred_labels_valid[class_mask]
            )
            class_accuracies[class_name] = float(class_acc)
        else:
            class_accuracies[class_name] = 0.0

    # Confusion matrix
    cm = confusion_matrix(true_labels_valid, pred_labels_valid, labels=[0, 1, 2])

    # Classification report
    class_names = ['Skin', 'Medicinal', 'Other']
    cr = classification_report(
        true_labels_valid,
        pred_labels_valid,
        target_names=class_names,
        output_dict=True
    )
    cr_text = classification_report(
        true_labels_valid,
        pred_labels_valid,
        target_names=class_names
    )

    # Confidence analysis
    correct_mask = true_labels_valid == pred_labels_valid
    correct_confidences = confidences_valid[correct_mask]
    incorrect_confidences = confidences_valid[~correct_mask]

    avg_confidence_correct = float(np.mean(correct_confidences)) if len(correct_confidences) > 0 else 0.0
    avg_confidence_incorrect = float(np.mean(incorrect_confidences)) if len(incorrect_confidences) > 0 else 0.0

    # High-confidence incorrect predictions (>90% confidence but wrong)
    high_conf_incorrect_mask = (~correct_mask) & (confidences_valid > 90.0)
    high_conf_incorrect_count = int(np.sum(high_conf_incorrect_mask))
    high_conf_incorrect_rate = float(np.sum(high_conf_incorrect_mask) / len(confidences_valid)) if len(confidences_valid) > 0 else 0.0

    # Gateway-specific safety metrics
    safety_metrics = {}
    # Skin incorrectly classified as Other
    skin_as_other = np.sum((true_labels_valid == 0) & (pred_labels_valid == 2))
    safety_metrics['Skin_as_Other'] = int(skin_as_other)
    safety_metrics['Skin_as_Other_rate'] = float(skin_as_other / np.sum(true_labels_valid == 0)) if np.sum(true_labels_valid == 0) > 0 else 0.0

    # Medicinal incorrectly classified as Other
    medicinal_as_other = np.sum((true_labels_valid == 1) & (pred_labels_valid == 2))
    safety_metrics['Medicinal_as_Other'] = int(medicinal_as_other)
    safety_metrics['Medicinal_as_Other_rate'] = float(medicinal_as_other / np.sum(true_labels_valid == 1)) if np.sum(true_labels_valid == 1) > 0 else 0.0

    # Other incorrectly classified as Skin
    other_as_skin = np.sum((true_labels_valid == 2) & (pred_labels_valid == 0))
    safety_metrics['Other_as_Skin'] = int(other_as_skin)
    safety_metrics['Other_as_Skin_rate'] = float(other_as_skin / np.sum(true_labels_valid == 2)) if np.sum(true_labels_valid == 2) > 0 else 0.0

    # Other incorrectly classified as Medicinal
    other_as_medicinal = np.sum((true_labels_valid == 2) & (pred_labels_valid == 1))
    safety_metrics['Other_as_Medicinal'] = int(other_as_medicinal)
    safety_metrics['Other_as_Medicinal_rate'] = float(other_as_medicinal / np.sum(true_labels_valid == 2)) if np.sum(true_labels_valid == 2) > 0 else 0.0

    # Prepare results dictionary
    results = {
        'overall_accuracy': float(overall_accuracy),
        'per_class_accuracy': class_accuracies,
        'confusion_matrix': cm.tolist(),
        'classification_report': cr,
        'confidence_analysis': {
            'avg_confidence_correct': avg_confidence_correct,
            'avg_confidence_incorrect': avg_confidence_incorrect,
            'high_confidence_incorrect_count': high_conf_incorrect_count,
            'high_confidence_incorrect_rate': high_conf_incorrect_rate,
            'total_samples': int(len(confidences_valid))
        },
        'safety_metrics': safety_metrics,
        'failed_predictions_count': len(failed_predictions_valid),
        'processing_errors': int(np.sum(~valid_mask)) if 'valid_mask' in locals() else 0
    }

    # Save results
    print("Saving results...")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. evaluation_report.json
    eval_report_path = RESULTS_DIR / 'evaluation_report.json'
    with open(eval_report_path, 'w') as f:
        json.dump(results, f, indent=2)

    # 2. classification_report.txt
    cr_path = RESULTS_DIR / 'classification_report.txt'
    with open(cr_path, 'w') as f:
        f.write(cr_text)

    # 3. confusion_matrix.png
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = RESULTS_DIR / 'confusion_matrix.png'
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # 4. failed_predictions.csv
    if len(failed_predictions_valid) > 0:
        failed_df = pd.DataFrame(failed_predictions_valid)
        failed_path = RESULTS_DIR / 'failed_predictions.csv'
        failed_df.to_csv(failed_path, index=False)
    else:
        # Create empty CSV with headers
        failed_df = pd.DataFrame(columns=['image_path', 'actual_class', 'predicted_class', 'confidence', 'error_type'])
        failed_path = RESULTS_DIR / 'failed_predictions.csv'
        failed_df.to_csv(failed_path, index=False)

    print(f"Evaluation complete. Results saved to {RESULTS_DIR}")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Skin Accuracy: {class_accuracies['Skin']:.4f}")
    print(f"Medicinal Accuracy: {class_accuracies['Medicinal']:.4f}")
    print(f"Other Accuracy: {class_accuracies['Other']:.4f}")
    print(f"Dangerous Skin->Other: {safety_metrics['Skin_as_Other']}")
    print(f"Dangerous Medicinal->Other: {safety_metrics['Medicinal_as_Other']}")
    print(f"High-confidence wrong predictions: {high_conf_incorrect_count}")

    return results


if __name__ == '__main__':
    evaluate_model()