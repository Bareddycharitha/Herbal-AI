import os
import pandas as pd


class HistoryLogger:

    def __init__(self, save_path):

        self.save_path = save_path

        self.history = []

    def log(
        self,
        epoch,
        train_metrics,
        val_metrics
    ):

        self.history.append({

            "Epoch": epoch,

            "Train Loss": train_metrics["loss"],

            "Train Accuracy": train_metrics["accuracy"],

            "Train Precision": train_metrics["precision"],

            "Train Recall": train_metrics["recall"],

            "Train F1": train_metrics["f1_score"],

            "Validation Loss": val_metrics["loss"],

            "Validation Accuracy": val_metrics["accuracy"],

            "Validation Precision": val_metrics["precision"],

            "Validation Recall": val_metrics["recall"],

            "Validation F1": val_metrics["f1_score"]

        })

    def save(self):

        os.makedirs(
            os.path.dirname(self.save_path),
            exist_ok=True
        )

        df = pd.DataFrame(self.history)

        df.to_csv(
            self.save_path,
            index=False
        )