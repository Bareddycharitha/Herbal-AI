import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from pathlib import Path

from ai.config import (
    DEVICE,
    TRAIN_DIR,
    TEST_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PATIENCE,
    MIN_DELTA,
    USE_AMP,
    CHECKPOINT_DIR,
    BEST_MODEL_PATH,
    LAST_MODEL_PATH,
    HISTORY_FILE,
    LABEL_SMOOTHING,
    USE_FOCAL_LOSS,
    FOCAL_GAMMA,
    CLASS_WEIGHTS,
    USE_OE,
    OE_DATASET_DIR,
    OE_RATIO,
    OE_LOSS_WEIGHT,
    CALIBRATE_AFTER_TRAINING,
    CALIBRATION_LR,
    CALIBRATION_MAX_ITER,
    ENSEMBLE_SIZE,
    ENSEMBLE_SEEDS,
    RANDOM_SEED,
    USE_TWO_STAGE,
    STAGE1_EPOCHS,
    STAGE2_EPOCHS,
    NUM_CLASSES,
    MODEL_NAME,
    PRETRAINED,
)

from ai.models.efficientnet import build_model
from ai.preprocessing.dataset import create_dataloaders, SkinDiseaseDataset
from ai.preprocessing.transforms import train_transform, test_transform
from ai.utils.early_stopping import EarlyStopping
from ai.utils.history import HistoryLogger
from ai.training.calibration import calibrate_model
from ai.training.ood_detection import EnergyBasedOOD, evaluate_ood_detector


# ==========================================================
# Focal Loss
# ==========================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
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
    """
    K = logits.size(1)
    uniform_target = torch.full_like(logits, 1.0 / K)
    log_probs = F.log_softmax(logits / temperature, dim=1)
    loss = F.kl_div(log_probs, uniform_target, reduction='batchmean', log_target=False)
    return loss


# ==========================================================
# Binary Dataset for Two-Stage Training
# ==========================================================

def create_binary_dataset(dataset, healthy_class_name="Unknown_Normal"):
    """
    Convert multi-class dataset to binary: Healthy (0) vs Diseased (1).
    """
    # Find healthy class index
    class_to_idx = dataset.class_to_idx
    healthy_idx = class_to_idx.get(healthy_class_name)

    if healthy_idx is None:
        raise ValueError(f"Healthy class '{healthy_class_name}' not found in dataset")

    # Create binary labels
    binary_samples = []
    for path, label in dataset.samples:
        binary_label = 0 if label == healthy_idx else 1
        binary_samples.append((path, binary_label))

    return binary_samples, healthy_idx


class BinarySkinDataset(torch.utils.data.Dataset):
    """Binary dataset wrapper: Healthy (0) vs Diseased (1)."""

    def __init__(self, base_dataset, healthy_idx, transform=None):
        self.base_dataset = base_dataset
        self.healthy_idx = healthy_idx
        self.transform = transform

        # Create binary samples
        self.samples = []
        for path, label in base_dataset.samples:
            binary_label = 0 if label == healthy_idx else 1
            self.samples.append((path, binary_label))

        self.classes = ["Healthy", "Diseased"]
        self.class_to_idx = {"Healthy": 0, "Diseased": 1}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, binary_label = self.samples[idx]
        image = self.base_dataset.loader(path)
        if self.transform:
            image = self.transform(image)
        return image, binary_label


# ==========================================================
# DataLoaders
# ==========================================================

def create_oe_dataset():
    """Create Outlier Exposure dataset."""
    if not OE_DATASET_DIR.exists():
        print(f"OE dataset not found at {OE_DATASET_DIR}, skipping OE")
        return None

    oe_dataset = SkinDiseaseDataset(
        root_dir=OE_DATASET_DIR,
        transform=train_transform,
    )
    print(f"Loaded OE dataset: {len(oe_dataset)} images")
    return oe_dataset


def create_dataloaders_with_oe():
    """Create train/val dataloaders with optional OE."""

    # Main training dataset
    train_dataset = SkinDiseaseDataset(
        root_dir=TRAIN_DIR,
        transform=train_transform,
    )

    val_dataset = SkinDiseaseDataset(
        root_dir=TEST_DIR,  # Using test as validation for now
        transform=test_transform,
    )

    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Class distribution
    train_labels = [label for _, label in train_dataset.samples]
    train_counts = Counter(train_labels)
    print("\nClass distribution:")
    for cls_idx, count in sorted(train_counts.items()):
        cls_name = train_dataset.classes[cls_idx]
        print(f"  {cls_name}: {count}")

    # Compute class weights
    if CLASS_WEIGHTS is None:
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_labels),
            y=train_labels
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
        print(f"\nComputed class weights: {class_weights.cpu().numpy()}")
    else:
        class_weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(DEVICE)
        print(f"\nUsing provided class weights: {class_weights.cpu().numpy()}")

    # Outlier Exposure
    oe_loader = None
    if USE_OE:
        oe_dataset = create_oe_dataset()
        if oe_dataset is not None:
            oe_loader = DataLoader(
                oe_dataset,
                batch_size=int(BATCH_SIZE * OE_RATIO),
                shuffle=True,
                num_workers=2,
                pin_memory=True,
                drop_last=True,
            )

    # Weighted sampler for balanced training
    weights = [1.0 / train_counts[label] for label in train_labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, oe_loader, class_weights, train_dataset.classes


def create_binary_dataloaders():
    """Create binary (Healthy vs Diseased) dataloaders."""
    train_dataset = SkinDiseaseDataset(
        root_dir=TRAIN_DIR,
        transform=train_transform,
    )

    val_dataset = SkinDiseaseDataset(
        root_dir=TEST_DIR,
        transform=test_transform,
    )

    # Get healthy class index
    healthy_idx = train_dataset.class_to_idx.get("Unknown_Normal")
    if healthy_idx is None:
        raise ValueError("Unknown_Normal class not found!")

    # Create binary datasets
    binary_train = BinarySkinDataset(train_dataset, healthy_idx, train_transform)
    binary_val = BinarySkinDataset(val_dataset, healthy_idx, test_transform)

    # Class weights for binary
    train_labels = [label for _, label in binary_train.samples]
    train_counts = Counter(train_labels)

    if CLASS_WEIGHTS is None:
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_labels),
            y=train_labels
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    else:
        # Use only first two weights for binary
        class_weights = torch.tensor(CLASS_WEIGHTS[:2], dtype=torch.float32).to(DEVICE)

    print(f"\nBinary Training: Healthy={train_counts[0]}, Diseased={train_counts[1]}")
    print(f"Binary Class weights: {class_weights.cpu().numpy()}")

    # Weighted sampler
    weights = [1.0 / train_counts[label] for label in train_labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        binary_train,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        binary_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, class_weights


# ==========================================================
# Training Functions
# ==========================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, oe_loader=None):
    model.train()

    loss_meter = AverageMeter()
    oe_loss_meter = AverageMeter()

    predictions = []
    targets = []

    progress = tqdm(loader, desc="Training", leave=False)
    oe_iter = iter(oe_loader) if oe_loader else None

    for images, labels in progress:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        with autocast(device_type="cuda", enabled=USE_AMP):
            outputs = model(images)
            loss = criterion(outputs, labels)

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
        if oe_loader:
            postfix['oe_loss'] = f"{oe_loss_meter.average:.4f}"
        progress.set_postfix(postfix)

    accuracy = accuracy_score(targets, predictions)
    macro_f1 = f1_score(targets, predictions, average='macro')

    return loss_meter.average, accuracy, macro_f1


def validate(model, loader, criterion):
    model.eval()

    loss_meter = AverageMeter()
    predictions = []
    targets = []
    all_probs = []

    with torch.no_grad():
        progress = tqdm(loader, desc="Validation", leave=False)
        for images, labels in progress:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            with autocast(device_type="cuda", enabled=USE_AMP):
                outputs = model(images)
                loss = criterion(outputs, labels)

            loss_meter.update(loss.item(), images.size(0))

            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            predictions.extend(preds.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu())

            progress.set_postfix(loss=f"{loss_meter.average:.4f}")

    accuracy = accuracy_score(targets, predictions)
    macro_f1 = f1_score(targets, predictions, average='macro')
    cm = confusion_matrix(targets, predictions)
    all_probs = torch.cat(all_probs, dim=0)

    return loss_meter.average, accuracy, macro_f1, cm, all_probs


class AverageMeter:
    """Tracks average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ==========================================================
# Main Training
# ==========================================================

def train_stage(model, train_loader, val_loader, criterion, optimizer, scheduler,
                scaler, epochs, stage_name, oe_loader=None):
    """Train for specified number of epochs."""
    history = []
    best_metric = 0.0
    early_stopping = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)

    print(f"\n{'='*60}")
    print(f"{stage_name} Training Started")
    print(f"{'='*60}")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, oe_loader
        )

        val_loss, val_acc, val_f1, val_cm, val_probs = validate(
            model, val_loader, criterion
        )

        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Macro F1: {train_f1:.4f}")
        print(f"Val Loss:   {val_loss:.4f}, Acc: {val_acc:.4f}, Macro F1: {val_f1:.4f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
        })

        # Save last checkpoint
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_accuracy": best_metric,
        }, LAST_MODEL_PATH)

        # Save best
        if val_acc > best_metric:
            best_metric = val_acc
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_accuracy": best_metric,
            }, BEST_MODEL_PATH)
            print("Best model updated.")

        if early_stopping(val_loss):
            print("Early stopping triggered.")
            break

    return history, best_metric


def train_multiclass(seed=RANDOM_SEED):
    """Train the 22-class disease classifier (Stage 2)."""
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, val_loader, oe_loader, class_weights, class_names = create_dataloaders_with_oe()

    model = build_model().to(DEVICE)

    # Loss
    if USE_FOCAL_LOSS:
        criterion = FocalLoss(gamma=FOCAL_GAMMA, weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    scaler = GradScaler(enabled=USE_AMP)

    epochs = STAGE2_EPOCHS if USE_TWO_STAGE else EPOCHS

    history, best_acc = train_stage(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        scaler, epochs, "Multi-class Disease", oe_loader
    )

    # Calibration
    if CALIBRATE_AFTER_TRAINING:
        print("\nCalibrating with Temperature Scaling...")
        checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])

        temp_scaler = calibrate_model(model, val_loader, DEVICE, CALIBRATION_LR, CALIBRATION_MAX_ITER)

        cal_path = CHECKPOINT_DIR / "temperature_scale.pth"
        torch.save({
            "temperature": temp_scaler.get_temperature(),
            "model_state_dict": model.state_dict(),
        }, cal_path)
        print(f"Calibration saved: {cal_path}, T={temp_scaler.get_temperature():.4f}")

    return model, class_names


def train_binary(seed=RANDOM_SEED):
    """Train binary Healthy vs Diseased classifier (Stage 1)."""
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, val_loader, class_weights = create_binary_dataloaders()

    # Binary model (2 classes)
    model = build_model(num_classes=2).to(DEVICE)

    criterion = FocalLoss(gamma=FOCAL_GAMMA, weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    scaler = GradScaler(enabled=USE_AMP)

    history, best_acc = train_stage(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        scaler, STAGE1_EPOCHS, "Binary Healthy/Diseased"
    )

    # Save binary model
    binary_path = CHECKPOINT_DIR / "best_binary_model.pth"
    torch.save(model.state_dict(), binary_path)
    print(f"Binary model saved to {binary_path}")

    return model


def train_two_stage():
    """Train two-stage pipeline: Binary -> Multi-class."""
    print("\n" + "="*60)
    print("TWO-STAGE TRAINING PIPELINE")
    print("="*60)

    # Stage 1: Binary classifier
    print("\n>>> Stage 1: Binary Healthy vs Diseased")
    binary_model = train_binary()

    # Stage 2: Multi-class classifier (diseases only)
    print("\n>>> Stage 2: 22-class Disease Classification")
    multiclass_model, class_names = train_multiclass()

    return binary_model, multiclass_model, class_names


def train_ensemble():
    """Train ensemble of models."""
    print(f"\nTraining ensemble of {ENSEMBLE_SIZE} models...")

    models = []
    for i, seed in enumerate(ENSEMBLE_SEEDS):
        print(f"\n{'='*60}")
        print(f"Ensemble Model {i+1}/{ENSEMBLE_SIZE} (seed={seed})")
        print(f"{'='*60}")

        if USE_TWO_STAGE:
            train_two_stage()
        else:
            train_multiclass(seed=seed)

        # Save ensemble checkpoint
        ensemble_path = CHECKPOINT_DIR / f"best_model_ensemble_{i}.pth"
        torch.save(torch.load(BEST_MODEL_PATH)["model_state_dict"], ensemble_path)
        print(f"Saved ensemble model {i} to {ensemble_path}")

    return models


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":
    if USE_TWO_STAGE:
        train_two_stage()
    else:
        train_multiclass()

    # Optionally train ensemble
    # train_ensemble()