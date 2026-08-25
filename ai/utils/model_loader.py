"""
Safe Model Checkpoint Loading Utilities

Provides consistent error handling, logging, and validation
for loading PyTorch model checkpoints across all inference modules.

Key guarantees:
- Verifies checkpoint file existence before loading
- Wraps torch.load() and model.load_state_dict() in exception handling
- Logs checkpoint path, model name/type, and exception details
- Raises ModelLoadError on any failure (never silently falls back to random weights)
- Supports both ``checkpoint["model_state_dict"]`` and direct state dict formats
- Exposes load status for readiness checks
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class ModelLoadStatus:
    """Tracks the load status of a model checkpoint."""
    model_name: str
    checkpoint_path: str
    loaded: bool = False
    error: Optional[str] = None
    error_type: Optional[str] = None
    missing_keys: list = field(default_factory=list)
    unexpected_keys: list = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        """Whether the model is loaded and ready for inference."""
        return self.loaded


class ModelLoadError(Exception):
    """Raised when a required model checkpoint fails to load.

    This exception is raised instead of silently falling back to
    randomly initialized weights, ensuring that model loading
    failures are never masked in production.
    """

    def __init__(self, model_name: str, checkpoint_path: Union[str, Path],
                 reason: str, original_exception: Optional[Exception] = None):
        self.model_name = model_name
        self.checkpoint_path = str(checkpoint_path)
        self.reason = reason
        self.original_exception = original_exception

        super().__init__(
            f"Failed to load model '{model_name}' from {checkpoint_path}: {reason}"
        )


def safe_load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    model_name: str = "unknown",
    remap_keys: Optional[Callable[[str], str]] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Safely load a model checkpoint with comprehensive error handling.

    Args:
        checkpoint_path: Path to the checkpoint file
        model: PyTorch model to load state dict into
        model_name: Human-readable name for logging
        remap_keys: Optional callable to remap state dict keys
        strict: Whether to use strict state_dict loading

    Returns:
        The full checkpoint dict (with metadata like epoch, accuracy, etc.)

    Raises:
        ModelLoadError: If the checkpoint file doesn't exist or loading fails
    """
    checkpoint_path = Path(checkpoint_path)

    # Step 1: Verify file exists
    if not checkpoint_path.exists():
        logger.error(
            "Checkpoint file not found",
            extra={
                "model_name": model_name,
                "path": str(checkpoint_path),
            },
        )
        raise ModelLoadError(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            reason="Checkpoint file not found",
        )

    # Step 2: Safely load the checkpoint file
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except FileNotFoundError as e:
        logger.error(
            "Checkpoint file disappeared during load",
            extra={
                "model_name": model_name,
                "path": str(checkpoint_path),
                "error_type": type(e).__name__,
                "error": str(e),
            },
        )
        raise ModelLoadError(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            reason=f"FileNotFoundError: {e}",
            original_exception=e,
        ) from e
    except Exception as e:
        logger.error(
            "Failed to load checkpoint file",
            extra={
                "model_name": model_name,
                "path": str(checkpoint_path),
                "error_type": type(e).__name__,
                "error": str(e),
            },
        )
        raise ModelLoadError(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            reason=f"{type(e).__name__}: {e}",
            original_exception=e,
        ) from e

    # Step 3: Extract state dict (supports both wrapped and direct formats)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        logger.info(
            "Extracted model_state_dict from checkpoint",
            extra={"model_name": model_name, "path": str(checkpoint_path)},
        )
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        logger.info(
            "Extracted state_dict from checkpoint",
            extra={"model_name": model_name, "path": str(checkpoint_path)},
        )
    else:
        state_dict = checkpoint

    # Step 4: Apply key remapping if provided
    if remap_keys is not None:
        original_keys = set(state_dict.keys())
        state_dict = {remap_keys(k): v for k, v in state_dict.items()}
        if set(state_dict.keys()) != original_keys:
            logger.info(
                "Applied key remapping for checkpoint compatibility",
                extra={
                    "model_name": model_name,
                    "path": str(checkpoint_path),
                    "original_key_count": len(original_keys),
                    "remapped_key_count": len(state_dict),
                },
            )

    # Step 5: Load state dict into model
    try:
        load_result = model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as e:
        logger.error(
            "State dict loading failed (architecture mismatch)",
            extra={
                "model_name": model_name,
                "path": str(checkpoint_path),
                "error_type": type(e).__name__,
                "error": str(e),
            },
        )
        raise ModelLoadError(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            reason=f"Architecture mismatch: {e}",
            original_exception=e,
        ) from e
    except Exception as e:
        logger.error(
            "Unexpected error loading state dict",
            extra={
                "model_name": model_name,
                "path": str(checkpoint_path),
                "error_type": type(e).__name__,
                "error": str(e),
            },
        )
        raise ModelLoadError(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            reason=f"{type(e).__name__}: {e}",
            original_exception=e,
        ) from e

    # Step 6: Log warnings for missing/unexpected keys
    missing, unexpected = load_result
    if missing:
        logger.warning(
            "Missing keys in state dict",
            extra={
                "model_name": model_name,
                "path": str(checkpoint_path),
                "missing_keys": missing,
            },
        )
    if unexpected:
            logger.warning(
                "Unexpected keys in state dict",
                extra={
                    "model_name": model_name,
                    "path": str(checkpoint_path),
                    "unexpected_keys": unexpected,
                },
            )

    logger.info(
        "Model loaded successfully",
        extra={
            "model_name": model_name,
            "path": str(checkpoint_path),
        },
    )

    return checkpoint


def check_checkpoint_status(
    checkpoint_path: Union[str, Path],
    model_name: str = "unknown",
) -> ModelLoadStatus:
    """
    Check if a checkpoint file exists and is loadable.

    Used for readiness checks to verify checkpoint availability
    without fully instantiating and loading a model.

    Args:
        checkpoint_path: Path to the checkpoint file
        model_name: Human-readable name for logging

    Returns:
        ModelLoadStatus with loaded=True if file exists and is loadable
    """
    status = ModelLoadStatus(
        model_name=model_name,
        checkpoint_path=str(checkpoint_path),
    )

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        status.error = f"Checkpoint file not found at {checkpoint_path}"
        status.error_type = "FileNotFoundError"
        return status

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        # Verify it's a valid checkpoint with state dict data
        if isinstance(checkpoint, dict):
            if "model_state_dict" not in checkpoint and "state_dict" not in checkpoint:
                # Direct state dict
                if not all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                    status.error = "Checkpoint does not contain valid state dict"
                    status.error_type = "ValueError"
                    return status
        elif not isinstance(checkpoint, dict):
            status.error = f"Unexpected checkpoint type: {type(checkpoint)}"
            status.error_type = "ValueError"
            return status

        status.loaded = True
        return status
    except Exception as e:
        status.error = f"{type(e).__name__}: {e}"
        status.error_type = type(e).__name__
        return status
