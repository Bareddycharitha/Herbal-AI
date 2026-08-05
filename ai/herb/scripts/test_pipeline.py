"""
Test complete Herb AI pipeline
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from herb.dataset import create_dataloaders
from herb.model import build_model


def main():

    train_loader, val_loader, test_loader, class_names, num_classes = (
        create_dataloaders()
    )

    model = build_model(num_classes)

    images, labels = next(iter(train_loader))

    print("=" * 60)
    print("Pipeline Working Successfully")
    print("=" * 60)
    print("Batch Shape :", images.shape)
    print("Labels Shape:", labels.shape)
    print("Classes     :", num_classes)
    print("=" * 60)


if __name__ == "__main__":
    main()