"""
Evaluate Herb Identification Model
"""

import sys
from pathlib import Path

import torch

# ==============================================================================
# Fix Project Path
# ==============================================================================

ROOT_DIR = Path(__file__).resolve().parents[3]

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# ==============================================================================
# Imports
# ==============================================================================

from ai.herb.dataset import create_dataloaders
from ai.herb.model import build_model
from ai.herb.training.trainer import HerbTrainer

from ai.herb.config import (
    DEVICE,
    BEST_MODEL_PATH,
    RESULTS_DIR,
)

from ai.mlflow_tracking import (
    start_evaluation_run,
    log_final_metrics,
    log_artifact_if_exists,
    set_tags,
    end_run,
)

# ==============================================================================
# Main
# ==============================================================================


def main():

    print("=" * 70)
    print("HERBAL-AI : HERB MODEL EVALUATION")
    print("=" * 70)

    # ==========================================================================
    # Start MLflow Evaluation Run
    # ==========================================================================

    start_evaluation_run(
        model_type="herb",
        run_name="Herb Classifier Test Evaluation",
    )

    set_tags({
        "dataset": "Medicinal_plant_dataset",
        "split": "test",
        "model_name": "tf_efficientnetv2_s",
    })

    try:

        train_loader, val_loader, test_loader, class_names, num_classes = (
            create_dataloaders()
        )

        model = build_model(num_classes)

        checkpoint = torch.load(
            BEST_MODEL_PATH,
            map_location=DEVICE,
        )

        # ======================================================================
        # Normalize checkpoint keys
        # Existing Herb checkpoint uses:
        #     model.*
        #
        # Current HerbClassifier expects:
        #     backbone.*
        #     classifier.*
        # ======================================================================

        state_dict = checkpoint["model_state_dict"]

        normalized_state_dict = {}

        for key, value in state_dict.items():

            if key.startswith("model.classifier."):
                new_key = key.replace(
                    "model.classifier.",
                    "classifier.",
                    1,
                )

            elif key.startswith("model."):
                new_key = key.replace(
                    "model.",
                    "backbone.",
                    1,
                )

            else:
                new_key = key

            normalized_state_dict[new_key] = value

        model.load_state_dict(normalized_state_dict)

        model.to(DEVICE)

        trainer = HerbTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            class_names=class_names,
        )

        loss, acc, metrics = trainer.validate()

        precision = metrics["precision"]
        recall = metrics["recall"]
        f1 = metrics["f1_score"]

        # ======================================================================
        # Log Test Metrics to MLflow
        # ======================================================================

        log_final_metrics({
            "test_loss": loss,
            "test_accuracy": acc,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1_score": f1,
        })

        # ======================================================================
        # Log Existing Evaluation Artifacts
        # ======================================================================

        log_artifact_if_exists(
            RESULTS_DIR / "classification_report.txt",
            artifact_path="evaluation",
        )

        log_artifact_if_exists(
            RESULTS_DIR / "confusion_matrix.png",
            artifact_path="evaluation",
        )

        log_artifact_if_exists(
            RESULTS_DIR / "evaluation_metrics.json",
            artifact_path="evaluation",
        )

        # ======================================================================
        # Print Results
        # ======================================================================

        print("\n" + "=" * 70)
        print("TEST SET RESULTS")
        print("=" * 70)

        print(f"Test Loss      : {loss:.4f}")
        print(f"Accuracy       : {acc:.2f}%")
        print(f"Precision      : {precision:.4f}")
        print(f"Recall         : {recall:.4f}")
        print(f"F1 Score       : {f1:.4f}")

        print("=" * 70)

    finally:

        # ======================================================================
        # End MLflow Run
        # ======================================================================

        end_run()


# ==============================================================================
# Run
# ==============================================================================

if __name__ == "__main__":
    main()