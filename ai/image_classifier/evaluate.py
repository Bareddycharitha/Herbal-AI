import json

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    ConfusionMatrixDisplay,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import (
    DEVICE,
    TEST_DIRS,
    BATCH_SIZE,
    NUM_WORKERS,
    BEST_MODEL_PATH,
    RESULTS_DIR,
)

from .dataset import UniversalImageDataset, LABEL_NAMES
from .model import build_model
from .transforms import test_transform


# ==========================================================
# Create Test DataLoader
# ==========================================================

def create_test_loader():

    test_dataset = UniversalImageDataset(
        dataset_dirs=TEST_DIRS,
        transform=test_transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Test Images : {len(test_dataset)}")

    return test_loader


# ==========================================================
# Load Model
# ==========================================================

def load_model():

    model = build_model()

    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    print("Best model loaded successfully.")

    return model


# ==========================================================
# Evaluate
# ==========================================================

def evaluate():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    test_loader = create_test_loader()

    model = load_model()

    predictions = []
    targets = []

    with torch.no_grad():

        progress = tqdm(
            test_loader,
            desc="Evaluating",
        )

        for images, labels in progress:

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True,
            )

            outputs = model(images)

            preds = outputs.argmax(dim=1)

            predictions.extend(
                preds.cpu().tolist()
            )

            targets.extend(
                labels.cpu().tolist()
            )

    print("\nUnique Ground Truth :", sorted(set(targets)))
    print("Unique Predictions  :", sorted(set(predictions)))

    labels = [0, 1, 2]
    class_names = [
        LABEL_NAMES[0],
        LABEL_NAMES[1],
        LABEL_NAMES[2],
    ]

    # ======================================================
    # Metrics
    # ======================================================

    accuracy = accuracy_score(
        targets,
        predictions,
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        targets,
        predictions,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        targets,
        predictions,
        labels=labels,
        target_names=class_names,
        zero_division=0,
    )

    cm = confusion_matrix(
        targets,
        predictions,
        labels=labels,
    )

    # ======================================================
    # Save Report
    # ======================================================

    with open(
        RESULTS_DIR / "classification_report.txt",
        "w",
    ) as f:
        f.write(report)

    # ======================================================
    # Save Metrics
    # ======================================================

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }

    with open(
        RESULTS_DIR / "evaluation_metrics.json",
        "w",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=4,
        )

    # ======================================================
    # Save Predictions
    # ======================================================

    with open(
        RESULTS_DIR / "test_predictions.csv",
        "w",
    ) as f:

        f.write("GroundTruth,Prediction\n")

        for gt, pred in zip(targets, predictions):

            f.write(
                f"{LABEL_NAMES[gt]},{LABEL_NAMES[pred]}\n"
            )

    # ======================================================
    # Confusion Matrix
    # ======================================================

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    fig, ax = plt.subplots(figsize=(7, 7))

    disp.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "confusion_matrix.png",
        dpi=300,
    )

    plt.close()

    # ======================================================
    # Console Output
    # ======================================================

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("=" * 60)

    print(report)

    print("\nResults saved to:")
    print(RESULTS_DIR)


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    evaluate()