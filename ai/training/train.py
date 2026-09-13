import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from PIL import Image

from ai.config import (
    DEVICE,
    TRAIN_DIR,
    VAL_SPLIT,
    NUM_WORKERS,
    PIN_MEMORY,
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
)

from ai.models.efficientnet import build_model
from ai.preprocessing.dataset import create_dataloaders
from ai.preprocessing.transforms import (
    get_train_transforms,
    get_valid_transforms,
)
from ai.training.calibration import calibrate_model


AMP_DEVICE = DEVICE.type


# ============================================================
# FOCAL LOSS
# ============================================================

class FocalLoss(nn.Module):

    def __init__(
        self,
        gamma=2.0,
        weight=None,
        label_smoothing=0.0,
    ):
        super().__init__()

        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):

        log_probs = F.log_softmax(
            inputs,
            dim=1,
        )

        log_pt = log_probs.gather(
            1,
            targets.unsqueeze(1),
        ).squeeze(1)

        pt = log_pt.exp()

        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )

        focal_loss = (
            (1.0 - pt) ** self.gamma
        ) * ce_loss

        return focal_loss.mean()


# ============================================================
# OUTLIER EXPOSURE LOSS
# ============================================================

def oe_loss(logits, temperature=1.0):

    num_classes = logits.size(1)

    uniform_target = torch.full_like(
        logits,
        1.0 / num_classes,
    )

    log_probs = F.log_softmax(
        logits / temperature,
        dim=1,
    )

    return F.kl_div(
        log_probs,
        uniform_target,
        reduction="batchmean",
        log_target=False,
    )


# ============================================================
# BINARY DATASET
# ============================================================

class BinarySkinDataset(
    torch.utils.data.Dataset
):

    def __init__(
        self,
        samples,
        healthy_idx,
        transform=None,
    ):

        self.healthy_idx = healthy_idx
        self.transform = transform

        self.samples = [
            (
                path,
                0 if label == healthy_idx else 1,
            )
            for path, label in samples
        ]

        self.classes = [
            "Healthy",
            "Diseased",
        ]

        self.class_to_idx = {
            "Healthy": 0,
            "Diseased": 1,
        }

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        path, binary_label = self.samples[idx]

        image = np.array(
            Image.open(path).convert("RGB")
        )

        if self.transform:

            image = self.transform(
                image=image
            )["image"]

        return image, binary_label


# ============================================================
# BINARY DATASET CREATION
# ============================================================

def create_binary_dataset(
    dataset,
    healthy_class_name="Unknown_Normal",
):

    healthy_idx = dataset.class_to_idx.get(
        healthy_class_name
    )

    if healthy_idx is None:

        raise ValueError(
            f"Healthy class '{healthy_class_name}' "
            "not found in dataset"
        )

    binary_samples = [
        (
            path,
            0 if label == healthy_idx else 1,
        )
        for path, label in dataset.samples
    ]

    return binary_samples, healthy_idx


# ============================================================
# OUTLIER DATASET
# ============================================================

def create_oe_dataset():

    if not OE_DATASET_DIR.exists():

        print(
            f"OE dataset not found at "
            f"{OE_DATASET_DIR}, skipping OE"
        )

        return None

    print(
        f"OE dataset found at "
        f"{OE_DATASET_DIR}, "
        "but OE is disabled for this experiment."
    )

    return None


# ============================================================
# DATALOADERS + CLASS WEIGHTS
# ============================================================

def create_dataloaders_with_oe():

    (
        train_loader,
        val_loader,
        class_names,
        test_loader,
    ) = create_dataloaders(TRAIN_DIR)

    train_dataset = train_loader.dataset

    train_labels = [
        label
        for _, label in train_dataset.samples
    ]

    train_counts = Counter(
        train_labels
    )

    print("\nClass distribution:")

    for cls_idx, count in sorted(
        train_counts.items()
    ):

        print(
            f"  {class_names[cls_idx]}: "
            f"{count}"
        )

    # --------------------------------------------------------
    # V4-A: Standard balanced class weights
    # --------------------------------------------------------

    if CLASS_WEIGHTS is None:

        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(train_labels),
            y=train_labels,
        )

        class_weights = torch.tensor(
            class_weights,
            dtype=torch.float32,
        ).to(DEVICE)

        print(
            "\nV4-A: Using standard "
            "balanced class weights:"
        )

        print(
            class_weights.cpu().numpy()
        )

    else:

        class_weights = torch.tensor(
            CLASS_WEIGHTS,
            dtype=torch.float32,
        ).to(DEVICE)

        print(
            "\nUsing configured class weights:"
        )

        print(
            class_weights.cpu().numpy()
        )

    # --------------------------------------------------------
    # OE disabled
    # --------------------------------------------------------

    oe_loader = None

    if USE_OE:

        oe_dataset = create_oe_dataset()

        if oe_dataset is not None:

            oe_loader = DataLoader(
                oe_dataset,
                batch_size=max(
                    1,
                    int(
                        BATCH_SIZE
                        * OE_RATIO
                    ),
                ),
                shuffle=True,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                drop_last=True,
            )

    return (
        train_loader,
        val_loader,
        oe_loader,
        class_weights,
        class_names,
    )


# ============================================================
# BINARY DATALOADERS
# ============================================================

def create_binary_dataloaders():

    full_dataset = ImageFolder(
        TRAIN_DIR
    )

    samples = full_dataset.samples

    labels = [
        label
        for _, label in samples
    ]

    healthy_idx = (
        full_dataset.class_to_idx.get(
            "Unknown_Normal"
        )
    )

    if healthy_idx is None:

        raise ValueError(
            "Unknown_Normal class not found "
            "in training dataset."
        )

    train_samples, val_samples = (
        train_test_split(
            samples,
            test_size=VAL_SPLIT,
            random_state=RANDOM_SEED,
            stratify=labels,
        )
    )

    binary_train = BinarySkinDataset(
        train_samples,
        healthy_idx,
        get_train_transforms(),
    )

    binary_val = BinarySkinDataset(
        val_samples,
        healthy_idx,
        get_valid_transforms(),
    )

    train_labels = [
        label
        for _, label
        in binary_train.samples
    ]

    train_counts = Counter(
        train_labels
    )

    print(
        f"\nBinary Training: "
        f"Healthy={train_counts[0]}, "
        f"Diseased={train_counts[1]}"
    )

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels),
        y=train_labels,
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
    ).to(DEVICE)

    print(
        "Binary Class weights: "
        f"{class_weights.cpu().numpy()}"
    )

    weights = [
        1.0 / train_counts[label]
        for label in train_labels
    ]

    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True,
    )

    train_loader = DataLoader(
        binary_train,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    val_loader = DataLoader(
        binary_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    return (
        train_loader,
        val_loader,
        class_weights,
    )


# ============================================================
# AVERAGE METER
# ============================================================

class AverageMeter:

    def __init__(self):

        self.reset()

    def reset(self):

        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):

        self.val = val

        self.sum += val * n

        self.count += n

        self.avg = (
            self.sum / self.count
        )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    oe_loader=None,
):

    model.train()

    loss_meter = AverageMeter()
    oe_loss_meter = AverageMeter()

    predictions = []
    targets = []

    progress = tqdm(
        loader,
        desc="Training",
        leave=False,
    )

    oe_iter = (
        iter(oe_loader)
        if oe_loader
        else None
    )

    for images, labels in progress:

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with autocast(
            device_type=AMP_DEVICE,
            enabled=USE_AMP,
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            if oe_iter is not None:

                try:

                    oe_images, _ = next(
                        oe_iter
                    )

                except StopIteration:

                    oe_iter = iter(
                        oe_loader
                    )

                    oe_images, _ = next(
                        oe_iter
                    )

                oe_images = oe_images.to(
                    DEVICE,
                    non_blocking=True,
                )

                oe_outputs = model(
                    oe_images
                )

                oe_l = oe_loss(
                    oe_outputs
                )

                loss = (
                    loss
                    + OE_LOSS_WEIGHT * oe_l
                )

                oe_loss_meter.update(
                    oe_l.item(),
                    oe_images.size(0),
                )

        scaler.scale(
            loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

        loss_meter.update(
            loss.item(),
            images.size(0),
        )

        preds = outputs.argmax(
            dim=1
        )

        predictions.extend(
            preds.detach()
            .cpu()
            .numpy()
        )

        targets.extend(
            labels.detach()
            .cpu()
            .numpy()
        )

        postfix = {
            "loss":
                f"{loss_meter.avg:.4f}"
        }

        if oe_loader:

            postfix["oe_loss"] = (
                f"{oe_loss_meter.avg:.4f}"
            )

        progress.set_postfix(
            postfix
        )

    accuracy = accuracy_score(
        targets,
        predictions,
    )

    macro_f1 = f1_score(
        targets,
        predictions,
        average="macro",
    )

    return (
        loss_meter.avg,
        accuracy,
        macro_f1,
    )


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader,
    criterion,
):

    model.eval()

    loss_meter = AverageMeter()

    predictions = []
    targets = []
    all_probs = []

    with torch.no_grad():

        progress = tqdm(
            loader,
            desc="Validation",
            leave=False,
        )

        for images, labels in progress:

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True,
            )

            with autocast(
                device_type=AMP_DEVICE,
                enabled=USE_AMP,
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels,
                )

            loss_meter.update(
                loss.item(),
                images.size(0),
            )

            probs = F.softmax(
                outputs,
                dim=1,
            )

            preds = outputs.argmax(
                dim=1
            )

            predictions.extend(
                preds.detach()
                .cpu()
                .numpy()
            )

            targets.extend(
                labels.detach()
                .cpu()
                .numpy()
            )

            all_probs.append(
                probs.detach()
                .cpu()
            )

            progress.set_postfix(
                loss=
                f"{loss_meter.avg:.4f}"
            )

    accuracy = accuracy_score(
        targets,
        predictions,
    )

    macro_f1 = f1_score(
        targets,
        predictions,
        average="macro",
    )

    cm = confusion_matrix(
        targets,
        predictions,
    )

    all_probs = torch.cat(
        all_probs,
        dim=0,
    )

    return (
        loss_meter.avg,
        accuracy,
        macro_f1,
        cm,
        all_probs,
    )


# ============================================================
# TRAINING STAGE
# ============================================================

def train_stage(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    epochs,
    stage_name,
    oe_loader=None,
):

    history = []

    best_metric = -float("inf")

    patience_counter = 0

    print(
        f"\n{'=' * 60}"
    )

    print(
        f"{stage_name} Training Started"
    )

    print(
        f"{'=' * 60}"
    )

    for epoch in range(epochs):

        print(
            f"\nEpoch "
            f"{epoch + 1}/{epochs}"
        )

        (
            train_loss,
            train_acc,
            train_f1,
        ) = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            oe_loader,
        )

        (
            val_loss,
            val_acc,
            val_f1,
            val_cm,
            val_probs,
        ) = validate(
            model,
            val_loader,
            criterion,
        )

        scheduler.step(
            val_loss
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        print(
            f"Train Loss: "
            f"{train_loss:.4f}, "
            f"Acc: {train_acc:.4f}, "
            f"Macro F1: {train_f1:.4f}"
        )

        print(
            f"Val Loss:   "
            f"{val_loss:.4f}, "
            f"Acc: {val_acc:.4f}, "
            f"Macro F1: {val_f1:.4f}"
        )

        print(
            f"Learning Rate: "
            f"{current_lr:.6f}"
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "train_macro_f1": train_f1,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
                "learning_rate": current_lr,
            }
        )

        # ----------------------------------------------------
        # Save latest checkpoint
        # ----------------------------------------------------

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "best_macro_f1":
                    best_metric,
            },
            LAST_MODEL_PATH,
        )

        # ----------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------

        if val_f1 > (
            best_metric + MIN_DELTA
        ):

            best_metric = val_f1

            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "best_macro_f1":
                        best_metric,
                },
                BEST_MODEL_PATH,
            )

            print(
                f"Best model updated! "
                f"Macro F1 = "
                f"{best_metric:.4f}"
            )

        else:

            patience_counter += 1

            print(
                "No Macro F1 improvement. "
                f"Patience: "
                f"{patience_counter}/{PATIENCE}"
            )

        if patience_counter >= PATIENCE:

            print(
                "\nEarly stopping triggered "
                f"after {patience_counter} "
                "epochs without Macro F1 "
                "improvement."
            )

            break

    return (
        history,
        best_metric,
    )


# ============================================================
# TRANSFER BINARY BACKBONE
# ============================================================

def transfer_binary_backbone(
    binary_model,
    multiclass_model,
):

    binary_state = (
        binary_model.state_dict()
    )

    multiclass_state = (
        multiclass_model.state_dict()
    )

    transferred = 0

    for key, value in binary_state.items():

        if key.startswith(
            "classifier."
        ):

            continue

        if (
            key in multiclass_state
            and multiclass_state[key].shape
            == value.shape
        ):

            multiclass_state[key] = value

            transferred += 1

    multiclass_model.load_state_dict(
        multiclass_state
    )

    print(
        f"\nTransferred {transferred} "
        "backbone parameters from "
        "Stage 1 → Stage 2."
    )

    return multiclass_model


# ============================================================
# MULTICLASS TRAINING
# ============================================================

def train_multiclass(
    binary_model=None,
    seed=RANDOM_SEED,
):

    print(
        f"Device: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    torch.manual_seed(seed)

    np.random.seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )

    (
        train_loader,
        val_loader,
        oe_loader,
        class_weights,
        class_names,
    ) = create_dataloaders_with_oe()

    model = build_model(
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    if binary_model is not None:

        print(
            "\nTransferring Stage 1 "
            "binary backbone to "
            "Stage 2 multiclass model..."
        )

        model = transfer_binary_backbone(
            binary_model,
            model,
        )

        print(
            "Stage 1 backbone transfer "
            "completed."
        )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    if USE_FOCAL_LOSS:

        criterion = FocalLoss(
            gamma=FOCAL_GAMMA,
            weight=class_weights,
            label_smoothing=LABEL_SMOOTHING,
        )

        print(
            "\nLoss: Focal Loss"
        )

    else:

        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=LABEL_SMOOTHING,
        )

        print(
            "\nLoss: Weighted CrossEntropy"
        )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print(
        f"Learning rate: "
        f"{LEARNING_RATE}"
    )

    print(
        f"Weight decay: "
        f"{WEIGHT_DECAY}"
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    # --------------------------------------------------------
    # AMP
    # --------------------------------------------------------

    scaler = GradScaler(
        enabled=USE_AMP
    )

    epochs = (
        STAGE2_EPOCHS
        if USE_TWO_STAGE
        else EPOCHS
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    history, best_f1 = train_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epochs=epochs,
        stage_name="Multi-class Disease",
        oe_loader=oe_loader,
    )

    print(
        f"\nBest validation "
        f"Macro F1: {best_f1:.4f}"
    )

    # --------------------------------------------------------
    # Load best model
    # --------------------------------------------------------

    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # --------------------------------------------------------
    # Temperature Calibration
    # --------------------------------------------------------

    if CALIBRATE_AFTER_TRAINING:

        print(
            "\nCalibrating with "
            "Temperature Scaling..."
        )

        temp_scaler = calibrate_model(
            model,
            val_loader,
            DEVICE,
            CALIBRATION_LR,
            CALIBRATION_MAX_ITER,
        )

        cal_path = (
            CHECKPOINT_DIR
            / "temperature_scale.pth"
        )

        torch.save(
            {
                "temperature":
                    temp_scaler.get_temperature(),
                "model_state_dict":
                    model.state_dict(),
            },
            cal_path,
        )

        print(
            f"Calibration saved: "
            f"{cal_path}"
        )

        print(
            f"Temperature: "
            f"{temp_scaler.get_temperature():.4f}"
        )

    return (
        model,
        class_names,
    )


# ============================================================
# BINARY TRAINING
# ============================================================

def train_binary(
    seed=RANDOM_SEED,
):

    print(
        f"Device: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    torch.manual_seed(seed)

    np.random.seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )

    (
        train_loader,
        val_loader,
        class_weights,
    ) = create_binary_dataloaders()

    model = build_model(
        num_classes=2
    ).to(DEVICE)

    criterion = FocalLoss(
        gamma=FOCAL_GAMMA,
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING,
    )

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

    scaler = GradScaler(
        enabled=USE_AMP
    )

    _, best_f1 = train_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epochs=STAGE1_EPOCHS,
        stage_name="Binary Healthy/Diseased",
    )

    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    binary_path = (
        CHECKPOINT_DIR
        / "best_binary_model.pth"
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),
            "best_macro_f1":
                best_f1,
        },
        binary_path,
    )

    print(
        f"\nBest binary model saved to: "
        f"{binary_path}"
    )

    return model


# ============================================================
# TWO-STAGE TRAINING
# ============================================================

def train_two_stage(
    seed=RANDOM_SEED,
):

    print(
        "\n" + "=" * 60
    )

    print(
        "STAGE 1: BINARY TRAINING"
    )

    print(
        "=" * 60
    )

    binary_model = train_binary(
        seed=seed
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "STAGE 2: MULTICLASS TRAINING"
    )

    print(
        "=" * 60
    )

    return train_multiclass(
        binary_model=binary_model,
        seed=seed,
    )


# ============================================================
# ENSEMBLE
# ============================================================

def train_ensemble():

    print(
        f"\nTraining ensemble of "
        f"{ENSEMBLE_SIZE} models..."
    )

    models = []

    for i, seed in enumerate(
        ENSEMBLE_SEEDS
    ):

        print(
            "\n" + "=" * 60
        )

        print(
            f"Ensemble Model "
            f"{i + 1}/{ENSEMBLE_SIZE} "
            f"(seed={seed})"
        )

        print(
            "=" * 60
        )

        if USE_TWO_STAGE:

            model, _ = train_two_stage(
                seed=seed
            )

        else:

            model, _ = train_multiclass(
                seed=seed
            )

        ensemble_path = (
            CHECKPOINT_DIR
            / f"best_model_ensemble_{i}.pth"
        )

        torch.save(
            model.state_dict(),
            ensemble_path,
        )

        print(
            f"Saved ensemble model "
            f"{i} to {ensemble_path}"
        )

        models.append(model)

    return models


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\n" + "=" * 70
    )

    print(
        "HERBAL-AI SKIN DISEASE MODEL"
    )

    print(
        "V4-A WINNER RECOVERY"
    )

    print(
        "=" * 70
    )

    print(
        "\nV4-A configuration:"
    )

    print(
        f"  Image size       : {320}"
    )

    print(
        f"  Batch size       : {BATCH_SIZE}"
    )

    print(
        f"  Epochs           : {EPOCHS}"
    )

    print(
        f"  Learning rate    : {LEARNING_RATE}"
    )

    print(
        f"  Weight decay     : {WEIGHT_DECAY}"
    )

    print(
        f"  Label smoothing  : "
        f"{LABEL_SMOOTHING}"
    )

    print(
        "  Class weighting  : "
        "Standard balanced"
    )

    print(
        "\nV4-A previously achieved:"
    )

    print(
        "  Test Accuracy    : 0.8034"
    )

    print(
        "  Test Macro F1    : 0.7755"
    )

    print(
        "\nThis run is only to "
        "recreate the selected "
        "V4-A checkpoint."
    )

    print(
        "No new experiment is "
        "being introduced."
    )

    if USE_TWO_STAGE:

        train_two_stage()

    else:

        train_multiclass()

    print(
        "\n" + "=" * 70
    )

    print(
        "V4-A RECOVERY TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )