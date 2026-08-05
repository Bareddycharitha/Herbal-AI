"""
Logging utilities for Herb Identification Training
"""

import logging
import sys

from ..config import RESULTS_DIR


def get_logger(name: str = "HerbTrainer"):
    """
    Creates and returns a configured logger.
    """

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)

    # Prevent duplicate handlers
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # File Handler
    file_handler = logging.FileHandler(
        RESULTS_DIR / "training.log",
        mode="w",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger