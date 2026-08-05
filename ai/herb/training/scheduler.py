"""
Learning Rate Schedulers for Herb Training
"""

from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    StepLR,
)

from ..config import (
    LR_SCHEDULER,
    COSINE_T_MAX,
    COSINE_ETA_MIN,
    LEARNING_RATE,
    EPOCHS,
)


def build_scheduler(optimizer, t_max=None, eta_min=None):
    """
    Build learning rate scheduler based on configuration.

    Args:
        optimizer: PyTorch optimizer
        t_max: Maximum iterations for cosine annealing
        eta_min: Minimum learning rate

    Returns:
        LR scheduler
    """
    if t_max is None:
        t_max = COSINE_T_MAX
    if eta_min is None:
        eta_min = COSINE_ETA_MIN

    if LR_SCHEDULER == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=eta_min,
        )
        print(f"Using CosineAnnealingLR (T_max={t_max}, eta_min={eta_min})")

    elif LR_SCHEDULER == "cosine_warm_restarts":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=t_max // 3,
            T_mult=2,
            eta_min=eta_min,
        )
        print(f"Using CosineAnnealingWarmRestarts")

    elif LR_SCHEDULER == "onecycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=LEARNING_RATE * 10,
            total_steps=EPOCHS,
            pct_start=0.3,
            anneal_strategy='cos',
        )
        print(f"Using OneCycleLR")

    elif LR_SCHEDULER == "step":
        scheduler = StepLR(
            optimizer,
            step_size=EPOCHS // 3,
            gamma=0.1,
        )
        print(f"Using StepLR")

    else:  # plateau (default)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.2,
            patience=2,
            min_lr=eta_min,
        )
        print(f"Using ReduceLROnPlateau")

    return scheduler