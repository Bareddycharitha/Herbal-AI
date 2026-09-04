"""
Universal Classifier Service

Wrapper for the universal image classifier with dependency injection support.
"""

from pathlib import Path
from typing import Optional

from fastapi import Depends
from backend.app.config import Settings, get_settings
from backend.app.exceptions import ModelLoadError
from backend.app.utils.logging import get_logger
from ai.utils.model_loader import ModelLoadStatus

logger = get_logger(__name__)

# Global instance for backward compatibility (will be replaced by DI)
_classifier_instance: Optional["UniversalClassifier"] = None


class UniversalClassifier:
    """
    Universal Image Classifier (3-class: Skin, Medicinal, Other)

    Uses the inference module with OOD detection and calibration.
    """

    def __init__(self, settings: Settings):
        # Import here to avoid circular imports
        from ai.image_classifier.inference import UniversalClassifierInference

        # Use the new inference class
        self.inference = UniversalClassifierInference(
            model_path=settings.universal_model_dir / "best_model.pth",
            use_calibration=True,
            calibration_path=settings.universal_model_dir / "temperature_scale.pth",
        )

        if self.inference.model is None:
            # The inference class swallows the load error and sets
            # self.model = None. We can't return a useful prediction in
            # that state — raise ModelLoadError so the request gets a
            # structured 503 instead of a confusing 500 AttributeError
            # ('NoneType' object has no attribute 'eval').
            checkpoint = settings.universal_model_dir / "best_model.pth"
            raise ModelLoadError(
                model_path=str(checkpoint),
                reason=self.inference.load_status.error or "checkpoint file is missing or unreadable",
            )

        logger.info(
            "Universal Image Classifier Loaded",
            calibration=True,
            model_path=str(settings.universal_model_dir / "best_model.pth"),
        )

    @property
    def model_load_status(self) -> ModelLoadStatus:
        """Expose model load status for readiness checks."""
        return self.inference.load_status

    @property
    def checkpoint_found(self) -> bool:
        """Check if universal classifier checkpoint file was present when initialized."""
        return getattr(self.inference, "checkpoint_found", True)

    def predict(self, image_path: str) -> dict:
        """
        Predict image class with OOD detection.

        Returns:
            dict with:
                - class: "Skin" | "Medicinal" | "Other"
                - confidence: float (0-100)
                - is_ood: bool
                - ood_scores: dict with energy, msp, entropy, combined
                - top_predictions: list of top-k predictions
        """
        if self.inference.model is None:
            # Belt-and-braces: the constructor raises, but the global
            # instance may have been replaced since. Re-check before
            # every call so we never hand back a half-baked result.
            raise ModelLoadError(
                model_path=str(self.inference.load_status.checkpoint_path),
                reason=self.inference.load_status.error or "checkpoint file is missing or unreadable",
            )
        result = self.inference.predict(image_path)

        return {
            "class": result["class"],
            "confidence": result["confidence"],
            "is_ood": result.get("is_ood", False),
            "ood_scores": result.get("ood_scores", {}),
            "top_predictions": result.get("top_predictions", []),
        }


def get_classifier(settings: Optional[Settings] = None) -> UniversalClassifier:
    """
    FastAPI dependency / accessor for getting the universal classifier.
    """
    global _classifier_instance
    if _classifier_instance is None:
        if settings is None or not isinstance(settings, Settings):
            settings = get_settings()
        _classifier_instance = UniversalClassifier(settings)
    return _classifier_instance


def init_classifier(settings: Settings) -> UniversalClassifier:
    """Initialize classifier at startup (for lifespan)."""
    global _classifier_instance
    _classifier_instance = UniversalClassifier(settings)
    return _classifier_instance


def shutdown_classifier() -> None:
    """Shutdown classifier (cleanup if needed)."""
    global _classifier_instance
    _classifier_instance = None


# Backward compatibility - deprecated
classifier = None  # Will be set by init_classifier()