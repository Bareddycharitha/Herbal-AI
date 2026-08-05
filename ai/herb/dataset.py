"""
Dataset utilities for Herb Identification
"""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

from .config import (
    DATASET_DIR,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
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


class TransformSubset(torch.utils.data.Dataset):
    """
    Applies different transforms to dataset subsets.
    """

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        image, label = self.subset[index]

        if self.transform:
            image = self.transform(image)

        return image, label


def create_dataloaders():
    """
    Creates train, validation and test dataloaders.
    """

    # -------------------------------------------------------
    # Load Dataset
    # -------------------------------------------------------

    full_dataset = ImageFolder(DATASET_DIR)

    class_names = full_dataset.classes

    num_classes = len(class_names)

    # -------------------------------------------------------
    # Save Class Mapping
    # -------------------------------------------------------

    class_mapping = {
        str(index): name
        for index, name in enumerate(class_names)
    }

    with open(CLASS_MAPPING_PATH, "w") as f:
        json.dump(class_mapping, f, indent=4)

    # -------------------------------------------------------
    # Dataset Split
    # -------------------------------------------------------

    total_size = len(full_dataset)

    train_size = int(TRAIN_RATIO * total_size)

    val_size = int(VAL_RATIO * total_size)

    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(RANDOM_SEED)

    train_subset, val_subset, test_subset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )

    # -------------------------------------------------------
    # Apply Transforms
    # -------------------------------------------------------

    train_dataset = TransformSubset(
        train_subset,
        train_transforms,
    )

    val_dataset = TransformSubset(
        val_subset,
        val_transforms,
    )

    test_dataset = TransformSubset(
        test_subset,
        test_transforms,
    )

    # -------------------------------------------------------
    # DataLoaders
    # -------------------------------------------------------

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

    print("=" * 60)
    print(" Herb Dataset Loaded Successfully")
    print("=" * 60)
    print(f"Total Images      : {total_size}")
    print(f"Training Images   : {train_size}")
    print(f"Validation Images : {val_size}")
    print(f"Testing Images    : {test_size}")
    print(f"Total Classes     : {num_classes}")
    print("=" * 60)

    return (
        train_loader,
        val_loader,
        test_loader,
        class_names,
        num_classes,
    )
