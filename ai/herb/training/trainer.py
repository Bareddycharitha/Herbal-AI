"""
Production Trainer for Herb Identification

Supports:
- ArcFace / Focal / CrossEntropy loss
- Outlier Exposure (OE)
- Mixed precision training
- Cosine annealing scheduler
- Early stopping
- Checkpointing
"""

import json
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import tqdm

from ..config import (
    DEVICE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    USE_AMP,
    HISTORY_FILE,
    TRAINING_CONFIG_PATH,
    MODEL_INFO_PATH,
    SAVE_EVERY,
    RESUME_TRAINING,
    PATIENCE,
    MIN_DELTA,
    LR_SCHEDULER,
    COSINE_T_MAX,
    COSINE_ETA_MIN,
    USE_OE,
    OE_LOSS_WEIGHT,
    CALIBRATE_AFTER_TRAINING,
    CALIBRATION_LR,
    CALIBRATION_MAX_ITER,
)

from .losses import build_loss
from .scheduler import build_scheduler
from .metrics import MetricsCalculator
from .checkpoint import CheckpointManager
from .visualization import TrainingVisualizer
from .logger import get_logger

from ai.training.calibration import calibrate_model


class HerbTrainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        class_names,
        oe_loader=None,
    ):

        self.device = DEVICE

        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.oe_loader = oe_loader

        self.class_names = class_names
        self.num_classes = len(class_names)

        # -----------------------------------------------------

        self.logger = get_logger()

        self.logger.info("=" * 60)
        self.logger.info("Initializing Herb Trainer")
        self.logger.info("=" * 60)

        # -----------------------------------------------------

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        # -----------------------------------------------------

        self.scheduler = build_scheduler(
            self.optimizer,
            t_max=COSINE_T_MAX,
            eta_min=COSINE_ETA_MIN,
        )

        # -----------------------------------------------------

        self.criteria = build_loss(
            train_loader.dataset,
            self.num_classes,
            self.device,
        )

        # -----------------------------------------------------

        self.scaler = GradScaler(
            "cuda",
            enabled=USE_AMP and self.device.type == "cuda",
        )

        # -----------------------------------------------------

        self.metrics = MetricsCalculator(
            class_names
        )

        self.checkpoint = CheckpointManager()

        self.visualizer = TrainingVisualizer()

        # -----------------------------------------------------

        self.best_accuracy = 0.0
        self.best_loss = float("inf")

        self.history = []
        self.start_epoch = 1

        self.training_start_time = None

        # -----------------------------------------------------

        if RESUME_TRAINING:
            self.logger.info("Loading previous checkpoint...")
            self.start_epoch = self.checkpoint.load(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
            )
            self.logger.info(f"Resuming from Epoch {self.start_epoch}")

        # -----------------------------------------------------

        self.save_training_configuration()

    # ==========================================================
    # Save Configuration
    # ==========================================================

    def save_training_configuration(self):
        config = {
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "device": str(self.device),
            "mixed_precision": USE_AMP,
            "number_of_classes": self.num_classes,
            "class_names": self.class_names,
            "use_arcface": "USE_ARCFACE" in globals() or True,  # Will be set from config
            "use_oe": USE_OE,
            "oe_loss_weight": OE_LOSS_WEIGHT,
        }

        with open(TRAINING_CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=4)

        self.logger.info("Training configuration saved.")

    # ==========================================================
    # Save Model Information
    # ==========================================================

    def save_model_information(self):
        total_params = sum(
            p.numel() for p in self.model.parameters()
        )
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        model_info = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_name": self.model.__class__.__name__,
        }

        with open(MODEL_INFO_PATH, "w") as f:
            json.dump(model_info, f, indent=4)

        self.logger.info("Model information saved.")

    # ==========================================================
    # Train One Epoch
    # ==========================================================

    def train_one_epoch(self, epoch):
        self.model.train()

        running_loss = 0.0
        running_cls_loss = 0.0
        running_oe_loss = 0.0

        correct = 0
        total = 0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{EPOCHS}",
            leave=False,
        )

        # OE iterator
        oe_iter = iter(self.oe_loader) if self.oe_loader else None

        for images, labels in progress_bar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            with autocast(
                device_type=self.device.type,
                enabled=USE_AMP,
            ):
                # Forward pass
                # For ArcFace, model needs labels during training
                if hasattr(self.model, 'use_arcface') and self.model.use_arcface:
                    outputs = self.model(images, labels)
                else:
                    outputs = self.model(images)

                # Classification loss
                cls_loss = self.criteria['cls'](outputs, labels)

                # Outlier Exposure loss
                oe_loss_val = 0.0
                if USE_OE and oe_iter is not None:
                    try:
                        oe_images, _ = next(oe_iter)
                    except StopIteration:
                        oe_iter = iter(self.oe_loader)
                        oe_images, _ = next(oe_iter)

                    oe_images = oe_images.to(self.device, non_blocking=True)

                    if hasattr(self.model, 'use_arcface') and self.model.use_arcface:
                        oe_outputs = self.model(oe_images)  # No labels for OE
                    else:
                        oe_outputs = self.model(oe_images)

                    oe_loss_val = self.criteria['oe'](oe_outputs)
                    cls_loss = cls_loss + OE_LOSS_WEIGHT * oe_loss_val

            self.scaler.scale(cls_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += cls_loss.item()
            running_cls_loss += self.criteria['cls'](outputs, labels).item()
            if USE_OE:
                running_oe_loss += oe_loss_val.item() if isinstance(oe_loss_val, torch.Tensor) else oe_loss_val

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            postfix = {'loss': f"{running_loss / (progress_bar.n + 1):.4f}"}
            if USE_OE:
                postfix['oe_loss'] = f"{running_oe_loss / (progress_bar.n + 1):.4f}"
            postfix['acc'] = f"{100 * correct / total:.2f}%"
            progress_bar.set_postfix(postfix)

        train_loss = running_loss / len(self.train_loader)
        train_accuracy = 100 * correct / total

        self.logger.info(f"Train Loss : {train_loss:.4f}")
        self.logger.info(f"Train Accuracy : {train_accuracy:.2f}%")

        return train_loss, train_accuracy

    # ==========================================================
    # Validation
    # ==========================================================

    @torch.no_grad()
    def validate(self):
        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        all_predictions = []
        all_labels = []

        progress_bar = tqdm(
            self.val_loader,
            desc="Validation",
            leave=False,
        )

        for images, labels in progress_bar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            with autocast(
                device_type=self.device.type,
                enabled=USE_AMP,
            ):
                if hasattr(self.model, 'use_arcface') and self.model.use_arcface:
                    outputs = self.model(images)  # No labels for inference
                else:
                    outputs = self.model(images)

                loss = self.criteria['cls'](outputs, labels)

            predictions = outputs.argmax(dim=1)

            running_loss += loss.item()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        val_loss = running_loss / len(self.val_loader)
        val_accuracy = 100 * correct / total

        metrics = self.metrics.calculate(all_labels, all_predictions)

        self.metrics.save_metrics(metrics)
        self.metrics.save_classification_report(all_labels, all_predictions)
        self.metrics.save_confusion_matrix(all_labels, all_predictions)

        self.logger.info(f"Validation Loss : {val_loss:.4f}")
        self.logger.info(f"Validation Accuracy : {val_accuracy:.2f}%")
        self.logger.info(f"Precision : {metrics['precision']:.4f}")
        self.logger.info(f"Recall : {metrics['recall']:.4f}")
        self.logger.info(f"F1 Score : {metrics['f1_score']:.4f}")

        return val_loss, val_accuracy, metrics

    # ==========================================================
    # Scheduler
    # ==========================================================

    def update_scheduler(self, validation_loss):
        if self.scheduler is not None:
            if LR_SCHEDULER == "plateau":
                self.scheduler.step(validation_loss)
            else:
                self.scheduler.step()

    # ==========================================================
    # Best Model Check
    # ==========================================================

    def is_best_model(self, validation_accuracy):
        return validation_accuracy > self.best_accuracy

    # ==========================================================
    # Save Checkpoints
    # ==========================================================

    def save_checkpoint(
        self,
        epoch,
        train_loss,
        val_loss,
        train_acc,
        val_acc,
    ):
        if val_acc > self.best_accuracy:
            self.best_accuracy = val_acc
            self.checkpoint.save_best(
                self.model,
                self.optimizer,
                self.scheduler,
                self.scaler,
                epoch,
                train_loss,
                val_loss,
                train_acc,
                val_acc,
            )
            self.logger.info("Best model saved.")

        self.checkpoint.save_last(
            self.model,
            self.optimizer,
            self.scheduler,
            self.scaler,
            epoch,
            train_loss,
            val_loss,
            train_acc,
            val_acc,
        )

        if epoch % SAVE_EVERY == 0:
            self.checkpoint.save_epoch(
                self.model,
                self.optimizer,
                self.scheduler,
                self.scaler,
                epoch,
                train_loss,
                val_loss,
                train_acc,
                val_acc,
            )
            self.logger.info(f"Epoch checkpoint saved ({epoch}).")

    # ==========================================================
    # Early Stopping
    # ==========================================================

    def early_stop(
        self,
        validation_loss,
        patience_counter,
        best_loss,
        patience,
        min_delta,
    ):
        if validation_loss < best_loss - min_delta:
            best_loss = validation_loss
            patience_counter = 0
        else:
            patience_counter += 1

        stop = patience_counter >= patience
        return stop, best_loss, patience_counter

    # ==========================================================
    # Train Model
    # ==========================================================

    def fit(self):
        self.logger.info("=" * 70)
        self.logger.info("Starting Herb Model Training")
        self.logger.info("=" * 70)

        self.save_model_information()
        self.training_start_time = time.time()

        patience = PATIENCE
        min_delta = MIN_DELTA
        patience_counter = 0
        best_loss = float("inf")

        for epoch in range(self.start_epoch, EPOCHS + 1):
            self.logger.info("")
            self.logger.info("-" * 70)
            self.logger.info(f"Epoch {epoch}/{EPOCHS}")
            self.logger.info("-" * 70)

            # --------------------------------------------------
            # Training
            # --------------------------------------------------
            train_loss, train_acc = self.train_one_epoch(epoch)

            # --------------------------------------------------
            # Validation
            # --------------------------------------------------
            val_loss, val_acc, metrics = self.validate()

            # --------------------------------------------------
            # Scheduler
            # --------------------------------------------------
            self.update_scheduler(val_loss)

            # --------------------------------------------------
            # Current Learning Rate
            # --------------------------------------------------
            current_lr = self.optimizer.param_groups[0]["lr"]

            # --------------------------------------------------
            # Save History
            # --------------------------------------------------
            history_row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1_score"],
                "learning_rate": current_lr,
            }
            self.history.append(history_row)

            # --------------------------------------------------
            # Save Checkpoints
            # --------------------------------------------------
            self.save_checkpoint(
                epoch, train_loss, val_loss, train_acc, val_acc
            )

            # --------------------------------------------------
            # Early Stopping
            # --------------------------------------------------
            stop, best_loss, patience_counter = self.early_stop(
                validation_loss=val_loss,
                patience_counter=patience_counter,
                best_loss=best_loss,
                patience=patience,
                min_delta=min_delta,
            )

            if stop:
                self.logger.info("")
                self.logger.info("Early stopping triggered.")
                break

        # ======================================================
        # Post-training Calibration
        # ======================================================
        if CALIBRATE_AFTER_TRAINING:
            self.logger.info("")
            self.logger.info("=" * 60)
            self.logger.info("Calibrating model with Temperature Scaling...")
            self.logger.info("=" * 60)

            # Load best model for calibration
            from ..config import BEST_MODEL_PATH
            checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
            self.model.load_state_dict(checkpoint["model_state_dict"])

            temp_scaler = calibrate_model(
                self.model,
                self.val_loader,
                DEVICE,
                lr=CALIBRATION_LR,
                max_iter=CALIBRATION_MAX_ITER,
            )

            # Save calibration temperature
            from ..config import CHECKPOINT_DIR
            cal_path = CHECKPOINT_DIR / "temperature_scale.pth"
            torch.save({
                "temperature": temp_scaler.get_temperature(),
                "model_state_dict": self.model.state_dict(),
            }, cal_path)

            self.logger.info(f"Calibration saved: {cal_path}, T={temp_scaler.get_temperature():.4f}")

        # ======================================================
        # Save Training History
        # ======================================================
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(HISTORY_FILE, index=False)

        # ======================================================
        # Generate Graphs
        # ======================================================
        self.visualizer.generate(self.history)

        # ======================================================
        # Training Time
        # ======================================================
        elapsed = time.time() - self.training_start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)

        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("Training Finished Successfully")
        self.logger.info("=" * 70)
        self.logger.info(f"Best Validation Accuracy : {self.best_accuracy:.2f}%")
        self.logger.info(f"Training Time : {hours}h {minutes}m {seconds}s")
        self.logger.info(f"History Saved : {HISTORY_FILE}")
        self.logger.info("=" * 70)

        return self.history