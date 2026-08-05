"""
Checkpoint Management for Herb Identification Model
"""

import torch
from pathlib import Path

from ..config import (
    CHECKPOINT_DIR,
    BEST_MODEL_PATH,
    LAST_MODEL_PATH,
)


class CheckpointManager:

    def __init__(self):
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------

    def save_best(
        self,
        model,
        optimizer,
        scheduler,
        scaler,
        epoch,
        train_loss,
        val_loss,
        train_acc,
        val_acc,
    ):
        """
        Save best model checkpoint.
        """

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_acc,
            "validation_accuracy": val_acc,
        }

        torch.save(
            checkpoint,
            BEST_MODEL_PATH,
        )

    # -----------------------------------------------------------------

    def save_last(
        self,
        model,
        optimizer,
        scheduler,
        scaler,
        epoch,
        train_loss,
        val_loss,
        train_acc,
        val_acc,
    ):
        """
        Save latest checkpoint.
        """

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_acc,
            "validation_accuracy": val_acc,
        }

        torch.save(
            checkpoint,
            LAST_MODEL_PATH,
        )

    # -----------------------------------------------------------------

    def save_epoch(
        self,
        model,
        optimizer,
        scheduler,
        scaler,
        epoch,
        train_loss,
        val_loss,
        train_acc,
        val_acc,
    ):
        """
        Save checkpoint every 5 epochs.
        """

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_acc,
            "validation_accuracy": val_acc,
        }

        save_path = CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pth"

        torch.save(
            checkpoint,
            save_path,
        )

    # -----------------------------------------------------------------

    def load(
        self,
        model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        checkpoint_path=LAST_MODEL_PATH,
    ):
        """
        Resume training from checkpoint.
        """

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        if optimizer is not None:
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        if (
            scheduler is not None
            and checkpoint["scheduler_state_dict"] is not None
        ):
            scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

        if (
            scaler is not None
            and checkpoint["scaler_state_dict"] is not None
        ):
            scaler.load_state_dict(
                checkpoint["scaler_state_dict"]
            )

        return checkpoint["epoch"] + 1