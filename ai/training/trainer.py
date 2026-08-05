import os
import torch
import torch.nn as nn
from tqdm import tqdm

from torch.amp import autocast, GradScaler

from ai.utils.metrics import calculate_metrics
from ai.utils.history import HistoryLogger
from ai.config import HISTORY_FILE
from ai.utils.early_stopping import EarlyStopping
from ai.config import PATIENCE, MIN_DELTA


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        epochs,
        checkpoint_dir
    ):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.history = HistoryLogger(HISTORY_FILE)
        self.device = device
        self.epochs = epochs
        self.early_stopping = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)

        self.scaler = GradScaler("cuda")

        self.best_accuracy = 0.0

        self.checkpoint_dir = checkpoint_dir

        os.makedirs(checkpoint_dir, exist_ok=True)

    def train_one_epoch(self):

        self.model.train()

        running_loss = 0

        predictions = []
        labels_list = []

        progress = tqdm(self.train_loader)

        for images, labels in progress:

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            with autocast(device_type="cuda"):

                outputs = self.model(images)

                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()

            self.scaler.step(self.optimizer)

            self.scaler.update()

            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            predictions.extend(preds.cpu().numpy())

            labels_list.extend(labels.cpu().numpy())

            progress.set_description(
                f"Train Loss: {loss.item():.4f}"
            )

        metrics = calculate_metrics(
            labels_list,
            predictions
        )

        metrics["loss"] = running_loss / len(self.train_loader)

        return metrics

    def validate(self):

        self.model.eval()

        running_loss = 0

        predictions = []
        labels_list = []

        with torch.no_grad():

            progress = tqdm(self.val_loader)

            for images, labels in progress:

                images = images.to(self.device)

                labels = labels.to(self.device)

                outputs = self.model(images)

                loss = self.criterion(outputs, labels)

                running_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)

                predictions.extend(preds.cpu().numpy())

                labels_list.extend(labels.cpu().numpy())

                progress.set_description(
                    f"Validation Loss: {loss.item():.4f}"
                )

        metrics = calculate_metrics(
            labels_list,
            predictions
        )

        metrics["loss"] = running_loss / len(self.val_loader)

        return metrics

    def save_checkpoint(self):

        torch.save(
            self.model.state_dict(),
            os.path.join(
                self.checkpoint_dir,
                "best_model.pth"
            )
        )

    def fit(self):

        for epoch in range(self.epochs):

            print("\n" + "=" * 60)
            print(f"Epoch {epoch+1}/{self.epochs}")
            print("=" * 60)

            train_metrics = self.train_one_epoch()

            val_metrics = self.validate()

            self.scheduler.step(val_metrics["loss"])

            print("\nTraining")

            print(train_metrics)

            print("\nValidation")

            self.history.log(
                epoch + 1,
                train_metrics,
                val_metrics
)

            self.history.save()

            self.early_stopping(val_metrics["loss"])

            if self.early_stopping.early_stop:

             print("\nEarly stopping activated.")

             break

            print(val_metrics)

            if val_metrics["accuracy"] > self.best_accuracy:

                self.best_accuracy = val_metrics["accuracy"]

                self.save_checkpoint()

                print("\n✅ Best Model Saved")