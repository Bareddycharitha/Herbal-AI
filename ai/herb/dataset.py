"""
Dataset utilities for Herb Identification
"""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from .config import (
    DATASET_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    CLASS_MAPPING_PATH,
)

from .transforms import (
    train_transforms,
    val_transforms,
    test_transforms,
)


class TransformDataset(torch.utils.data.Dataset):
    """
    Applies a specific transform to an ImageFolder dataset.
    """

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]

        if self.transform:
            image = self.transform(image)

        return image, label


def create_dataloaders():
    """
    Creates train, validation and test dataloaders
    from the existing dataset splits.
    """

    # ==========================================================================
    # Dataset Directories
    # ==========================================================================

    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "val"
    test_dir = DATASET_DIR / "test"

    # ==========================================================================
    # Validate Dataset Structure
    # ==========================================================================

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training directory not found: {train_dir}"
        )

    if not val_dir.exists():
        raise FileNotFoundError(
            f"Validation directory not found: {val_dir}"
        )

    if not test_dir.exists():
        raise FileNotFoundError(
            f"Testing directory not found: {test_dir}"
        )

    # ==========================================================================
    # Load Each Split
    # ==========================================================================

    train_base = ImageFolder(train_dir)
    val_base = ImageFolder(val_dir)
    test_base = ImageFolder(test_dir)

    # ==========================================================================
    # Verify Class Consistency
    # ==========================================================================

    train_classes = train_base.classes
    val_classes = val_base.classes
    test_classes = test_base.classes

    if train_classes != val_classes:
        raise ValueError(
            "Training and validation classes do not match.\n"
            f"Training classes: {train_classes}\n"
            f"Validation classes: {val_classes}"
        )

    if train_classes != test_classes:
        raise ValueError(
            "Training and testing classes do not match.\n"
            f"Training classes: {train_classes}\n"
            f"Testing classes: {test_classes}"
        )

    class_names = train_classes
    num_classes = len(class_names)

    # ==========================================================================
    # Verify Expected Number Of Classes
    # ==========================================================================

    print("\nDetected herb classes:")
    for index, class_name in enumerate(class_names):
        print(f"  {index}: {class_name}")

    print(f"\nTotal herb classes detected: {num_classes}")

    # ==========================================================================
    # Save Class Mapping
    # ==========================================================================

    class_mapping = {
        str(index): name
        for index, name in enumerate(class_names)
    }

    with open(CLASS_MAPPING_PATH, "w") as f:
        json.dump(
            class_mapping,
            f,
            indent=4,
        )

    # ==========================================================================
    # Apply Transforms
    # ==========================================================================

    train_dataset = TransformDataset(
        train_base,
        train_transforms,
    )

    val_dataset = TransformDataset(
        val_base,
        val_transforms,
    )

    test_dataset = TransformDataset(
        test_base,
        test_transforms,
    )

    # ==========================================================================
    # DataLoaders
    # ==========================================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    # ==========================================================================
    # Dataset Information
    # ==========================================================================

    print("=" * 60)
    print(" Herb Dataset Loaded Successfully")
    print("=" * 60)

    print(f"Dataset Directory : {DATASET_DIR}")
    print(f"Training Images   : {len(train_dataset)}")
    print(f"Validation Images : {len(val_dataset)}")
    print(f"Testing Images    : {len(test_dataset)}")
    print(f"Total Classes     : {num_classes}")

    print("=" * 60)

    return (
        train_loader,
        val_loader,
        test_loader,
        class_names,
        num_classes,
    )