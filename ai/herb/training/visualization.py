"""
Visualization utilities for Herb Identification Training
"""

from pathlib import Path
import matplotlib.pyplot as plt

from ..config import RESULTS_DIR


class TrainingVisualizer:

    def __init__(self):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # Loss Curve
    # ----------------------------------------------------------

    def plot_loss_curve(self, history):

        epochs = [x["epoch"] for x in history]

        train_loss = [x["train_loss"] for x in history]

        val_loss = [x["val_loss"] for x in history]

        plt.figure(figsize=(10, 6))

        plt.plot(
            epochs,
            train_loss,
            label="Training Loss",
            linewidth=2,
        )

        plt.plot(
            epochs,
            val_loss,
            label="Validation Loss",
            linewidth=2,
        )

        plt.title("Training vs Validation Loss")

        plt.xlabel("Epoch")

        plt.ylabel("Loss")

        plt.grid(True)

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            RESULTS_DIR / "loss_curve.png",
            dpi=300,
        )

        plt.close()

    # ----------------------------------------------------------
    # Accuracy Curve
    # ----------------------------------------------------------

    def plot_accuracy_curve(self, history):

        epochs = [x["epoch"] for x in history]

        train_acc = [x["train_acc"] for x in history]

        val_acc = [x["val_acc"] for x in history]

        plt.figure(figsize=(10, 6))

        plt.plot(
            epochs,
            train_acc,
            label="Training Accuracy",
            linewidth=2,
        )

        plt.plot(
            epochs,
            val_acc,
            label="Validation Accuracy",
            linewidth=2,
        )

        plt.title("Training vs Validation Accuracy")

        plt.xlabel("Epoch")

        plt.ylabel("Accuracy (%)")

        plt.grid(True)

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            RESULTS_DIR / "accuracy_curve.png",
            dpi=300,
        )

        plt.close()

    # ----------------------------------------------------------
    # Learning Rate Curve
    # ----------------------------------------------------------

    def plot_learning_rate(self, history):

        if "learning_rate" not in history[0]:
            return

        epochs = [x["epoch"] for x in history]

        learning_rates = [
            x["learning_rate"]
            for x in history
        ]

        plt.figure(figsize=(10, 6))

        plt.plot(
            epochs,
            learning_rates,
            linewidth=2,
        )

        plt.title("Learning Rate Schedule")

        plt.xlabel("Epoch")

        plt.ylabel("Learning Rate")

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            RESULTS_DIR / "learning_rate_curve.png",
            dpi=300,
        )

        plt.close()

    # ----------------------------------------------------------
    # Training Summary
    # ----------------------------------------------------------

    def save_dashboard(self, history):

        if not history:
            return

        last = history[-1]

        summary_path = RESULTS_DIR / "training_dashboard.txt"

        with open(summary_path, "w") as f:

            f.write("=" * 60 + "\n")

            f.write("HERBAL-AI TRAINING SUMMARY\n")

            f.write("=" * 60 + "\n\n")

            f.write(f"Epochs Completed : {last['epoch']}\n")

            f.write(f"Train Loss       : {last['train_loss']:.4f}\n")

            f.write(f"Validation Loss  : {last['val_loss']:.4f}\n")

            f.write(f"Train Accuracy   : {last['train_acc']:.2f}%\n")

            f.write(f"Validation Acc   : {last['val_acc']:.2f}%\n")

            f.write(f"Precision        : {last['precision']:.4f}\n")

            f.write(f"Recall           : {last['recall']:.4f}\n")

            f.write(f"F1 Score         : {last['f1']:.4f}\n")

            f.write("\n")

            f.write("=" * 60)

    # ----------------------------------------------------------

    def generate(self, history):

        self.plot_loss_curve(history)

        self.plot_accuracy_curve(history)

        self.plot_learning_rate(history)

        self.save_dashboard(history)