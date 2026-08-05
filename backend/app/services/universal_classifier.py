"""
Universal Classifier Service

Wrapper for the universal image classifier with dependency injection support.
"""

from pathlib import Path
from typing import Optional

from fastapi import Depends
from backend.app.config import Settings, get_settings


# Global instance for backward compatibility (will be replaced by DI)
_classifier_instance: Optional["UniversalClassifier"] = None


class UniversalClassifier:
    """
    Universal Image Classifier (3-class: Skin, Medicinal, Other)

    Uses the inference module with ensemble, OOD detection, and calibration.
    """

    def __init__(self, settings: Settings):
        # Import here to avoid circular imports
        from ai.image_classifier.inference import UniversalClassifierInference

        # Use the new inference class
        self.inference = UniversalClassifierInference(
            model_path=settings.universal_model_dir / "best_model.pth",
            use_calibration=True,
            calibration_path=settings.universal_model_dir / "temperature_scale.pth",
            ood_threshold=0.5,
        )

        print("Universal Image Classifier Loaded (with OOD detection & calibration)")

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
        result = self.inference.predict(image_path)

        return {
            "class": result["class"],
            "confidence": result["confidence"],
            "is_ood": result.get("is_ood", False),
            "ood_scores": result.get("ood_scores", {}),
            "top_predictions": result.get("top_predictions", []),
        }


def get_classifier(settings: Settings = Depends(get_settings)) -> UniversalClassifier:
    """
    FastAPI dependency for getting the universal classifier.

    Creates a new instance per request (or use singleton via lru_cache if needed).
    For production, consider using a singleton with proper lifecycle management.
    """
    global _classifier_instance
    if _classifier_instance is None:
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