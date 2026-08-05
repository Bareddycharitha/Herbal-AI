import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from ai.config import (
    DEVICE,
    TRAIN_DIR,
    CHECKPOINT_DIR,
    RESULTS_DIR,
)

from ai.preprocessing.dataset import create_dataloaders
from ai.models.efficientnet import build_model


def evaluate():

    # ----------------------------------
    # Create Results Folder
    # ----------------------------------
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ----------------------------------
    # Load Dataset
    # ----------------------------------
    train_loader, val_loader, class_names, test_loader = create_dataloaders(
        TRAIN_DIR
    )

    # ----------------------------------
    # Load Model
    # ----------------------------------
    model = build_model().to(DEVICE)

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        "best_model.pth"
    )

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=DEVICE
        )
    )

    model.eval()

    print("=" * 60)
    print("Model Loaded Successfully")
    print("=" * 60)

    predictions = []
    labels = []

    # ----------------------------------
    # Prediction Loop
    # ----------------------------------
    with torch.no_grad():

        for images, target in test_loader:

            images = images.to(DEVICE)
            target = target.to(DEVICE)

            outputs = model(images)

            preds = torch.argmax(outputs, dim=1)

            predictions.extend(
                preds.cpu().numpy()
            )

            labels.extend(
                target.cpu().numpy()
            )

    predictions = np.array(predictions)
    labels = np.array(labels)

    # ----------------------------------
    # Metrics
    # ----------------------------------
    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    print("\n")
    print("=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    # ----------------------------------
    # Classification Report
    # ----------------------------------
    report = classification_report(
        labels,
        predictions,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    report_path = os.path.join(
        RESULTS_DIR,
        "classification_report.txt"
    )

    with open(report_path, "w") as f:
        f.write(report)

    print("\nClassification Report Saved")

    # ----------------------------------
    # Confusion Matrix
    # ----------------------------------
    cm = confusion_matrix(
        labels,
        predictions
    )

    fig, ax = plt.subplots(figsize=(16, 16))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot(
        cmap="Blues",
        ax=ax,
        xticks_rotation=90,
        colorbar=False
    )

    plt.tight_layout()

    cm_path = os.path.join(
        RESULTS_DIR,
        "confusion_matrix.png"
    )

    plt.savefig(
        cm_path,
        dpi=300
    )

    plt.close()

    print("Confusion Matrix Saved")

    # ----------------------------------
    # Save Metrics
    # ----------------------------------
    metrics = {

        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)

    }

    metrics_path = os.path.join(
        RESULTS_DIR,
        "evaluation_metrics.json"
    )

    with open(metrics_path, "w") as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    print("Metrics Saved")

    print("=" * 60)
    print("Evaluation Completed Successfully")
    print("=" * 60)


if __name__ == "__main__":

    evaluate()