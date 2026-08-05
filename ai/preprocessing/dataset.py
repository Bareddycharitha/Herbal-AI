import numpy as np
from PIL import Image

from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder

from ai.preprocessing.transforms import (
    get_train_transforms,
    get_valid_transforms
)

from ai.config import (
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    VAL_SPLIT,
    RANDOM_SEED,
    TEST_DIR
)


class SkinDiseaseDataset(Dataset):

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


def create_dataloaders(train_dir):

    # -----------------------------
    # Train Dataset
    # -----------------------------
    dataset = ImageFolder(train_dir)

    class_names = dataset.classes

    samples = dataset.samples

    labels = [sample[1] for sample in samples]

    # -----------------------------
    # Train / Validation Split
    # -----------------------------
    train_samples, val_samples = train_test_split(
        samples,
        test_size=VAL_SPLIT,
        random_state=RANDOM_SEED,
        stratify=labels
    )

    train_dataset = SkinDiseaseDataset(
        train_samples,
        transform=get_train_transforms()
    )

    val_dataset = SkinDiseaseDataset(
        val_samples,
        transform=get_valid_transforms()
    )

    # -----------------------------
# Test Dataset
# -----------------------------
    test_folder = ImageFolder(TEST_DIR)

    test_dataset = SkinDiseaseDataset(
        test_folder.samples,
        transform=get_valid_transforms()
    )
    # -----------------------------
    # DataLoaders
    # -----------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    return train_loader, val_loader, class_names, test_loader