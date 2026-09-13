"""
Skin Disease Model - Test Evaluation

Evaluates the current best skin disease model on the held-out TEST set.

Uses existing project files only.
Saves the canonical evaluation artifacts in ai/results/.
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tqdm import tqdm

from ai.config import (
    BEST_MODEL_PATH,
    DEVICE,
    NUM_CLASSES,
    TRAIN_DIR,
)
from ai.preprocessing.dataset import create_dataloaders
from ai.models.efficientnet import build_model


def load_checkpoint(model):
    """
    Load the best trained checkpoint into the model.

    Supports the checkpoint formats already used by the project.
    """
    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE,
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    return checkpoint


def main():
    print("=" * 60)
    print("SKIN DISEASE TEST EVALUATION")
    print("=" * 60)

    # ---------------------------------------------------------
    # DATA
    # ---------------------------------------------------------
    train_loader, val_loader, class_names, test_loader = (
        create_dataloaders(TRAIN_DIR)
    )

    print("\n" + "=" * 60)
    print("SKIN DISEASE DATASET")
    print("=" * 60)

    print(f"Training samples   : {len(train_loader.dataset)}")
    print(f"Validation samples : {len(val_loader.dataset)}")
    print(f"Testing samples    : {len(test_loader.dataset)}")
    print(f"Number of classes  : {len(class_names)}")

    print("\nClasses:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")

    # ---------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("LOADING MODEL")
    print("=" * 60)

    # IMPORTANT:
    # build_model() only accepts num_classes.
    # PRETRAINED is handled internally by ai.config.
    model = build_model(
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    checkpoint = load_checkpoint(model)

    model.eval()

    print("Model Loaded Successfully")

    if isinstance(checkpoint, dict):
        if "epoch" in checkpoint:
            print(
                f"Best checkpoint epoch : "
                f"{checkpoint['epoch']}"
            )

        if "best_metric" in checkpoint:
            print(
                f"Stored best Macro F1 : "
                f"{checkpoint['best_metric']:.4f}"
            )

        if "val_f1" in checkpoint:
            print(
                f"Stored validation F1 : "
                f"{checkpoint['val_f1']:.4f}"
            )

    # ---------------------------------------------------------
    # TEST EVALUATION
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("RUNNING TEST EVALUATION")
    print("=" * 60)

    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for images, targets in tqdm(
            test_loader,
            desc="Testing",
        ):
            images = images.to(DEVICE)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1,
            )

            all_targets.extend(
                targets.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    y_true = np.asarray(all_targets)
    y_pred = np.asarray(all_predictions)

    # ---------------------------------------------------------
    # METRICS
    # ---------------------------------------------------------
    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
    )

    # ---------------------------------------------------------
    # PRINT RESULTS
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    print(f"Accuracy           : {accuracy:.4f}")
    print(f"Macro Precision    : {macro_precision:.4f}")
    print(f"Macro Recall       : {macro_recall:.4f}")
    print(f"Macro F1           : {macro_f1:.4f}")
    print(f"Weighted Precision : {weighted_precision:.4f}")
    print(f"Weighted Recall    : {weighted_recall:.4f}")
    print(f"Weighted F1        : {weighted_f1:.4f}")

    print("\nClassification Report:")
    print(report)

    # ---------------------------------------------------------
    # RESULTS DIRECTORY
    # ---------------------------------------------------------
    results_dir = Path("ai") / "results"
    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # CLASSIFICATION REPORT
    # ---------------------------------------------------------
    report_path = (
        results_dir / "classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(report)

    # ---------------------------------------------------------
    # CONFUSION MATRIX
    # ---------------------------------------------------------
    cm_path = (
        results_dir / "confusion_matrix.png"
    )

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(
            figsize=(16, 14)
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )

        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(
            "Skin Disease Test Confusion Matrix"
        )

        plt.tight_layout()

        plt.savefig(
            cm_path,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close()

    except Exception as e:
        print(
            f"\nWarning: Could not save "
            f"confusion matrix: {e}"
        )

    # ---------------------------------------------------------
    # METRICS JSON
    # ---------------------------------------------------------
    metrics = {
        "dataset": "test",
        "num_samples": int(len(y_true)),
        "num_classes": int(len(class_names)),
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(
            weighted_precision
        ),
        "weighted_recall": float(
            weighted_recall
        ),
        "weighted_f1": float(
            weighted_f1
        ),
    }

    metrics_path = (
        results_dir / "evaluation_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=4,
        )

    # ---------------------------------------------------------
    # FINAL OUTPUT
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST ARTIFACTS SAVED")
    print("=" * 60)

    print(report_path.resolve())
    print(cm_path.resolve())
    print(metrics_path.resolve())

    print("=" * 60)


if __name__ == "__main__":
    main()