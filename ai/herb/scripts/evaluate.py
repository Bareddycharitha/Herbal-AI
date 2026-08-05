"""
Evaluate Herb Identification Model
"""

import sys
from pathlib import Path

import torch

# ==============================================================================
# Fix Project Path
# ==============================================================================

ROOT_DIR = Path(__file__).resolve().parents[3]

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# ==============================================================================
# Imports
# ==============================================================================

from ai.herb.dataset import create_dataloaders
from ai.herb.model import build_model
from ai.herb.training.trainer import HerbTrainer

from ai.herb.config import (
    DEVICE,
    BEST_MODEL_PATH,
)

# ==============================================================================
# Main
# ==============================================================================


def main():

    print("=" * 70)
    print("HERBAL-AI : HERB MODEL EVALUATION")
    print("=" * 70)

    train_loader, val_loader, test_loader, class_names, num_classes = (
        create_dataloaders()
    )

    model = build_model(num_classes)

    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)

    trainer = HerbTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        class_names=class_names,
    )
    loss, acc, metrics = trainer.validate()

    precision = metrics["precision"]
    recall = metrics["recall"]
    f1 = metrics["f1_score"]
    

    print("\n" + "=" * 70)
    print("TEST SET RESULTS")
    print("=" * 70)

    print(f"Test Loss      : {loss:.4f}")
    print(f"Accuracy       : {acc:.2f}%")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1 Score       : {f1:.4f}")

    print("=" * 70)


# ==============================================================================
# Run
# ==============================================================================

if __name__ == "__main__":
    main()