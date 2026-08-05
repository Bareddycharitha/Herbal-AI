import os
import random
import csv

import numpy as np
import torch

from .config import RANDOM_SEED


# ==========================================================
# Random Seed
# ==========================================================

def set_seed(seed=RANDOM_SEED):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================================
# Average Meter
# ==========================================================

class AverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n

    @property
    def average(self):
        if self.count == 0:
            return 0
        return self.sum / self.count


# ==========================================================
# Save Checkpoint
# ==========================================================

def save_checkpoint(state, filename):

    torch.save(state, filename)

    print(f"Checkpoint saved -> {filename}")


# ==========================================================
# Load Checkpoint
# ==========================================================

def load_checkpoint(model, optimizer, filename, device):

    checkpoint = torch.load(
        filename,
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint["epoch"]
    best_accuracy = checkpoint["best_accuracy"]

    print(f"Checkpoint loaded from epoch {epoch}")

    return model, optimizer, epoch, best_accuracy


# ==========================================================
# Early Stopping
# ==========================================================

class EarlyStopping:

    def __init__(self, patience=5, min_delta=0.001):

        self.patience = patience
        self.min_delta = min_delta

        self.best_loss = float("inf")
        self.counter = 0

    def __call__(self, validation_loss):

        if validation_loss < self.best_loss - self.min_delta:

            self.best_loss = validation_loss
            self.counter = 0

            return False

        self.counter += 1

        print(
            f"EarlyStopping: "
            f"{self.counter}/{self.patience}"
        )

        return self.counter >= self.patience


# ==========================================================
# Save Training History
# ==========================================================

def save_history(history, csv_path):

    if len(history) == 0:
        return

    keys = history[0].keys()

    with open(csv_path, "w", newline="") as file:

        writer = csv.DictWriter(file, fieldnames=keys)

        writer.writeheader()

        writer.writerows(history)

    print(f"Training history saved -> {csv_path}")