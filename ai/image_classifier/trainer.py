import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.amp import autocast, GradScaler

from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from .config import (
    DEVICE,
    TRAIN_DIRS,
    VAL_DIRS,
    BATCH_SIZE,
    NUM_WORKERS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    PATIENCE,
    MIN_DELTA,
    USE_AMP,
    BEST_MODEL_PATH,
    LAST_MODEL_PATH,
    TRAIN_HISTORY,
    LABEL_SMOOTHING,
    USE_FOCAL_LOSS,
    FOCAL_GAMMA,
    CLASS_WEIGHTS,
    USE_OE,
    OE_DATASET_DIR,
    OE_RATIO,
    OE_LOSS_WEIGHT,
    USE_HARD_NEGATIVES,
    HARD_NEGATIVE_DIR,
    HARD_NEGATIVE_RATIO,
    CALIBRATE_AFTER_TRAINING,
    CALIBRATION_LR,
    CALIBRATION_MAX_ITER,
    ENSEMBLE_SIZE,
    ENSEMBLE_SEEDS,
    PIN_MEMORY,
    RANDOM_SEED,
    IMAGE_SIZE,
)

from .dataset import UniversalImageDataset
from .transforms import train_transform, test_transform
from .model import build_model
from .utils import (
    set_seed,
    AverageMeter,
    EarlyStopping,
    save_checkpoint,
    save_history,
)

from ai.training.calibration import calibrate_model
from ai.training.ood_detection import EnergyBasedOOD

from collections import Counter
from pathlib import Path


# ==========================================================
# Focal Loss
# ==========================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Where p_t is the predicted probability for the true class.
    """

    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ==========================================================
# Outlier Exposure Loss
# ==========================================================

def oe_loss(logits, temperature=1.0):
    """
    Outlier Exposure loss: encourages uniform predictions on OOD data.

    L_OE = -1/K * sum(log(softmax(logits/T)))

    This is equivalent to maximizing entropy of predictions on OOD samples.
    """
    # Uniform target distribution
    K = logits.size(1)
    uniform_target = torch.full_like(logits, 1.0 / K)

    # KL divergence between model predictions and uniform
    log_probs = F.log_softmax(logits / temperature, dim=1)
    loss = F.kl_div(log_probs, uniform_target, reduction='batchmean', log_target=False)

    return loss


# ==========================================================
# DataLoaders with Hard Negatives & OE
# ==========================================================

def create_oe_dataset():
    """Create Outlier Exposure dataset from diverse images."""
    if not OE_DATASET_DIR.exists():
        print(f"OE dataset not found at {OE_DATASET_DIR}, skipping OE")
        return None

    oe_dataset = UniversalImageDataset(
        dataset_dirs=[OE_DATASET_DIR],
        transform=train_transform,
    )
    print(f"Loaded OE dataset: {len(oe_dataset)} images")
    return oe_dataset


def create_hard_negative_dataset():
    """Create hard negative dataset for 'Other' class."""
    if not HARD_NEGATIVE_DIR.exists():
        print(f"Hard negative dataset not found at {HARD_NEGATIVE_DIR}, skipping")
        return None

    hard_neg_dataset = UniversalImageDataset(
        dataset_dirs=[HARD_NEGATIVE_DIR],
        transform=train_transform,
    )
    print(f"Loaded hard negative dataset: {len(hard_neg_dataset)} images")
    return hard_neg_dataset


def create_dataloaders():
    """Create train and validation dataloaders with optional OE and hard negatives."""

    # Main training dataset
    train_dataset = UniversalImageDataset(
        dataset_dirs=TRAIN_DIRS,
        transform=train_transform,
    )

    val_dataset = UniversalImageDataset(
        dataset_dirs=VAL_DIRS,
        transform=test_transform,
    )

    # Dataset Statistics
    train_counts = Counter(label for _, label in train_dataset.samples)

    print("\n" + "=" * 60)
    print("TRAIN DATASET DISTRIBUTION")
    print("=" * 60)
    print(f"Skin       : {train_counts[0]}")
    print(f"Medicinal  : {train_counts[1]}")
    print(f"Other      : {train_counts[2]}")

    val_counts = Counter(label for _, label in val_dataset.samples)

    print("\n" + "=" * 60)
    print("VALIDATION DATASET DISTRIBUTION")
    print("=" * 60)
    print(f"Skin       : {val_counts[0]}")
    print(f"Medicinal  : {val_counts[1]}")
    print(f"Other      : {val_counts[2]}")
    print("=" * 60)

    # Compute class weights
    if CLASS_WEIGHTS is None:
        labels = [label for _, label in train_dataset.samples]
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels),
            y=labels
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
        print(f"\nComputed class weights: {class_weights.cpu().numpy()}")
    else:
        class_weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(DEVICE)
        print(f"\nUsing provided class weights: {class_weights.cpu().numpy()}")

    # Optional: Hard Negative Mining
    if USE_HARD_NEGATIVES:
        hard_neg_dataset = create_hard_negative_dataset()
        if hard_neg_dataset is not None:
            # Combine with main dataset
            train_dataset = ConcatDataset([train_dataset, hard_neg_dataset])
            print(f"Combined train dataset size: {len(train_dataset)}")

    # Optional: Outlier Exposure
    oe_loader = None
    if USE_OE:
        oe_dataset = create_oe_dataset()
        if oe_dataset is not None:
            oe_loader = DataLoader(
                oe_dataset,
                batch_size=int(BATCH_SIZE * OE_RATIO),
                shuffle=True,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                drop_last=True,
            )
            print(f"OE loader created with batch size: {int(BATCH_SIZE * OE_RATIO)}")

    # Create weighted sampler for balanced batches
    if hasattr(train_dataset, 'samples'):
        # Single dataset
        labels = [label for _, label in train_dataset.samples]
    else:
        # ConcatDataset
        labels = []
        for dataset in train_dataset.datasets:
            labels.extend([label for _, label in dataset.samples])

    # WeightedRandomSampler for balanced sampling
    class_counts = Counter(labels)
    weights = [1.0 / class_counts[label] for label in labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    print(f"\nTraining Images   : {len(train_dataset)}")
    print(f"Validation Images : {len(val_dataset)}\n")

    return train_loader, val_loader, oe_loader, class_weights


# ==========================================================
# Train One Epoch
# ==========================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    oe_loader=None,
    epoch=0,
):
    model.train()

    loss_meter = AverageMeter()
    oe_loss_meter = AverageMeter()

    predictions = []
    targets = []

    progress = tqdm(loader, desc="Training", leave=False)

    # Create OE iterator
    oe_iter = iter(oe_loader) if oe_loader is not None else None

    for images, labels in progress:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        with autocast(
            device_type="cuda",
            enabled=USE_AMP,
        ):
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Outlier Exposure loss
            if oe_iter is not None:
                try:
                    oe_images, _ = next(oe_iter)
                except StopIteration:
                    oe_iter = iter(oe_loader)
                    oe_images, _ = next(oe_iter)

                oe_images = oe_images.to(DEVICE, non_blocking=True)
                oe_outputs = model(oe_images)
                oe_l = oe_loss(oe_outputs)
                loss = loss + OE_LOSS_WEIGHT * oe_l
                oe_loss_meter.update(oe_l.item(), oe_images.size(0))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss.item(), images.size(0))

        preds = outputs.argmax(dim=1)
        predictions.extend(preds.cpu().numpy())
        targets.extend(labels.cpu().numpy())

        postfix = {'loss': f"{loss_meter.average:.4f}"}
        if oe_loader is not None:
            postfix['oe_loss'] = f"{oe_loss_meter.average:.4f}"
        progress.set_postfix(postfix)

    accuracy = accuracy_score(targets, predictions)
    macro_f1 = f1_score(targets, predictions, average='macro')

    return loss_meter.average, accuracy, macro_f1


# ==========================================================
# Validation
# ==========================================================

def validate(
    model,
    loader,
    criterion,
):
    model.eval()

    loss_meter = AverageMeter()

    predictions = []
    targets = []

    with torch.no_grad():
        progress = tqdm(loader, desc="Validation", leave=False)

        for images, labels in progress:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            with autocast(
                device_type="cuda",
                enabled=USE_AMP,
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)

            loss_meter.update(loss.item(), images.size(0))

            preds = outputs.argmax(dim=1)
            predictions.extend(preds.cpu().numpy())
            targets.extend(labels.cpu().numpy())

            progress.set_postfix(loss=f"{loss_meter.average:.4f}")

    accuracy = accuracy_score(targets, predictions)
    macro_f1 = f1_score(targets, predictions, average='macro')
    cm = confusion_matrix(targets, predictions)

    return loss_meter.average, accuracy, macro_f1, cm


# ==========================================================
# Training Engine
# ==========================================================

def train(seed=RANDOM_SEED):
    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU")

    set_seed(seed)

    train_loader, val_loader, oe_loader, class_weights = create_dataloaders()

    model = build_model().to(DEVICE)

    # Loss function
    if USE_FOCAL_LOSS:
        criterion = FocalLoss(
            gamma=FOCAL_GAMMA,
            weight=class_weights,
            label_smoothing=LABEL_SMOOTHING,
        )
        print(f"Using Focal Loss (gamma={FOCAL_GAMMA}, label_smoothing={LABEL_SMOOTHING})")
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=LABEL_SMOOTHING,
        )
        print(f"Using CrossEntropy Loss (label_smoothing={LABEL_SMOOTHING})")

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    scaler = GradScaler(enabled=USE_AMP)

    early_stopping = EarlyStopping(
        patience=PATIENCE,
        min_delta=MIN_DELTA,
    )

    best_accuracy = 0.0
    best_macro_f1 = 0.0

    history = []

    print("\nTraining Started\n")

    for epoch in range(EPOCHS):
        print("=" * 60)
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print("=" * 60)

        train_loss, train_acc, train_f1 = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            oe_loader=oe_loader,
            epoch=epoch,
        )

        val_loss, val_acc, val_f1, val_cm = validate(
            model,
            val_loader,
            criterion,
        )

        scheduler.step(val_loss)

        print(f"Train Loss : {train_loss:.4f}")
        print(f"Train Acc  : {train_acc:.4f}")
        print(f"Train Macro F1: {train_f1:.4f}")
        print(f"Val Loss   : {val_loss:.4f}")
        print(f"Val Acc    : {val_acc:.4f}")
        print(f"Val Macro F1 : {val_f1:.4f}")
        print(f"Val Confusion Matrix:\n{val_cm}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
        })

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_accuracy": best_accuracy,
                "best_macro_f1": best_macro_f1,
            },
            LAST_MODEL_PATH,
        )

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_macro_f1 = val_f1

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_accuracy": best_accuracy,
                    "best_macro_f1": best_macro_f1,
                },
                BEST_MODEL_PATH,
            )

            print("Best model updated.")

        if early_stopping(val_loss):
            print("Early stopping triggered.")
            break

    save_history(history, TRAIN_HISTORY)

    print("\nTraining Finished")
    print(f"Best Validation Accuracy : {best_accuracy:.4f}")
    print(f"Best Validation Macro F1 : {best_macro_f1:.4f}")

    # ==========================================================
    # Post-training Calibration
    # ==========================================================
    if CALIBRATE_AFTER_TRAINING:
        print("\n" + "=" * 60)
        print("Calibrating model with Temperature Scaling...")
        print("=" * 60)

        # Load best model for calibration
        checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])

        temp_scaler = calibrate_model(
            model,
            val_loader,
            DEVICE,
            lr=CALIBRATION_LR,
            max_iter=CALIBRATION_MAX_ITER,
        )

        # Save calibration temperature
        cal_path = BEST_MODEL_PATH.parent / "temperature_scale.pth"
        torch.save({
            "temperature": temp_scaler.get_temperature(),
            "model_state_dict": model.state_dict(),
        }, cal_path)

        print(f"Calibration temperature saved to {cal_path}")
        print(f"Learned temperature: {temp_scaler.get_temperature():.4f}")

    return model


# ==========================================================
# Ensemble Training
# ==========================================================

def train_ensemble():
    """Train ensemble of models with different seeds."""
    print(f"\nTraining ensemble of {ENSEMBLE_SIZE} models...")
    models = []

    for i, seed in enumerate(ENSEMBLE_SEEDS):
        print(f"\n{'='*60}")
        print(f"Training ensemble model {i+1}/{ENSEMBLE_SIZE} (seed={seed})")
        print(f"{'='*60}")

        model = train(seed=seed)

        # Save ensemble model
        ensemble_path = BEST_MODEL_PATH.parent / f"best_model_ensemble_{i}.pth"
        torch.save(model.state_dict(), ensemble_path)
        models.append(model)
        print(f"Saved ensemble model {i} to {ensemble_path}")

    return models


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":
    

    # Train single model
    model = train()

    # Optionally train ensemble
    # models = train_ensemble()