import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight


def get_class_weights(train_loader, device):
    """
    Compute balanced class weights from the training dataset.
    """

    labels = []

    for _, batch_labels in train_loader:
        labels.extend(batch_labels.numpy())

    labels = np.array(labels)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32
    ).to(device)

    return class_weights