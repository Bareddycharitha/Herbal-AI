"""
Train the Herb Identification Model
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from ai.herb.dataset import create_dataloaders
from ai.herb.model import build_model
from ai.herb.training.trainer import HerbTrainer


def main():

    print("=" * 70)
    print("HERBAL-AI : HERB IDENTIFICATION TRAINING")
    print("=" * 70)

    train_loader, val_loader, test_loader, class_names, num_classes = (
        create_dataloaders()
    )

    model = build_model(num_classes)

    trainer = HerbTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_names=class_names,
    )

    trainer.fit()

    print()
    print("Training Completed Successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()