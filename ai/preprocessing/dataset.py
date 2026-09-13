"""
Dataset utilities for Skin Disease Classification
"""

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split

from ai.preprocessing.transforms import (
    get_train_transforms,
    get_valid_transforms,
)

from ai.config import (
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    VAL_SPLIT,
    RANDOM_SEED,
    TEST_DIR,
)


# ==============================================================================
# Dataset
# ==============================================================================

class SkinDiseaseDataset(Dataset):
    """
    Dataset wrapper for skin disease images.

    samples:
        List of (image_path, class_index)
    """

    def __init__(self, samples, transform=None):

        self.samples = samples
        self.transform = transform

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        image = np.array(image)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label


# ==============================================================================
# DataLoaders
# ==============================================================================

def create_dataloaders(train_dir):
    """
    Create train, validation and test dataloaders.

    IMPORTANT:
        Validation is created only from the training dataset.

        TEST_DIR is kept completely separate and is used only
        for final evaluation.
    """

    # ==========================================================================
    # Load Training Dataset
    # ==========================================================================

    dataset = ImageFolder(train_dir)

    class_names = dataset.classes

    samples = dataset.samples

    labels = [sample[1] for sample in samples]

    # ==========================================================================
    # Train / Validation Split
    # ==========================================================================

    train_samples, val_samples = train_test_split(
        samples,
        test_size=VAL_SPLIT,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    # ==========================================================================
    # Create Train Dataset
    # ==========================================================================

    train_dataset = SkinDiseaseDataset(
        train_samples,
        transform=get_train_transforms(),
    )

    # ==========================================================================
    # Create Validation Dataset
    # ==========================================================================

    val_dataset = SkinDiseaseDataset(
        val_samples,
        transform=get_valid_transforms(),
    )

    # ==========================================================================
    # Create Independent Test Dataset
    # ==========================================================================

    test_folder = ImageFolder(TEST_DIR)

    # Verify class ordering
    if test_folder.classes != class_names:
        raise ValueError(
            "Training and testing classes do not match.\n\n"
            f"Training classes: {class_names}\n"
            f"Testing classes:  {test_folder.classes}"
        )

    test_dataset = SkinDiseaseDataset(
        test_folder.samples,
        transform=get_valid_transforms(),
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

    print("\n" + "=" * 60)
    print("SKIN DISEASE DATASET")
    print("=" * 60)

    print(f"Training samples   : {len(train_dataset)}")
    print(f"Validation samples : {len(val_dataset)}")
    print(f"Testing samples    : {len(test_dataset)}")
    print(f"Number of classes  : {len(class_names)}")

    print("\nClasses:")

    for index, class_name in enumerate(class_names):
        print(f"  {index}: {class_name}")

    print("=" * 60)

    return (
        train_loader,
        val_loader,
        class_names,
        test_loader,
    )