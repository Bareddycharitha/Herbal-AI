"""
Evaluation Metrics for Herb Identification
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from ..config import (
    RESULTS_DIR,
    METRICS_PATH,
)


class MetricsCalculator:

    def __init__(self, class_names):

        self.class_names = class_names

    # -------------------------------------------------------------

    def calculate(self, labels, predictions):
        """
        Calculate all evaluation metrics.
        """

        metrics = {
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision_score(
                labels,
                predictions,
                average="weighted",
                zero_division=0,
            ),
            "recall": recall_score(
                labels,
                predictions,
                average="weighted",
                zero_division=0,
            ),
            "f1_score": f1_score(
                labels,
                predictions,
                average="weighted",
                zero_division=0,
            ),
        }

        return metrics

    # -------------------------------------------------------------

    def save_metrics(self, metrics):
        """
        Save metrics to JSON.
        """

        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=4)

    # -------------------------------------------------------------

    def save_classification_report(
        self,
        labels,
        predictions,
    ):
        """
        Save classification report.
        """

        report = classification_report(
            labels,
            predictions,
            target_names=self.class_names,
            zero_division=0,
        )

        report_path = RESULTS_DIR / "classification_report.txt"

        with open(report_path, "w") as f:
            f.write(report)

    # -------------------------------------------------------------

    def save_confusion_matrix(
        self,
        labels,
        predictions,
    ):
        """
        Save confusion matrix.
        """

        cm = confusion_matrix(
            labels,
            predictions,
        )

        fig = plt.figure(figsize=(16, 16))

        plt.imshow(cm, interpolation="nearest")

        plt.title("Confusion Matrix")

        plt.colorbar()

        tick_marks = np.arange(len(self.class_names))

        plt.xticks(
            tick_marks,
            self.class_names,
            rotation=90,
            fontsize=7,
        )

        plt.yticks(
            tick_marks,
            self.class_names,
            fontsize=7,
        )

        plt.xlabel("Predicted")

        plt.ylabel("Actual")

        plt.tight_layout()

        plt.savefig(
            RESULTS_DIR / "confusion_matrix.png",
            dpi=300,
        )

        plt.close(fig)