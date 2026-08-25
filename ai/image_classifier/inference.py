"""
Universal Image Classifier Inference

Provides ensemble inference, OOD detection, and calibrated predictions.
"""

import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional, Union
import numpy as np

from .config import (
    DEVICE,
    BEST_MODEL_PATH,
    IMAGE_SIZE,
    NUM_CLASSES,
    MODEL_NAME,
    ENSEMBLE_SIZE,
    ENSEMBLE_SEEDS,
    OOD_ENERGY_THRESHOLD,
    OOD_MSP_THRESHOLD,
    OOD_ENTROPY_THRESHOLD,
    CALIBRATION_PATH,
)

from .model import build_model
from .transforms import test_transform, get_tta_transforms

from ai.training.calibration import ModelWithTemperature, compute_ece
from ai.training.ood_detection import EnergyBasedOOD, MSPBasedOOD, EntropyBasedOOD, CombinedOODDetector
from ai.utils.model_loader import safe_load_checkpoint, ModelLoadError, ModelLoadStatus


CLASS_NAMES = {
    0: "Skin",
    1: "Medicinal",
    2: "Other",
}

CLASS_NAMES_REV = {v: k for k, v in CLASS_NAMES.items()}


class UniversalClassifierInference:
    """
    Universal Image Classifier with ensemble, OOD detection, and calibration.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        ensemble_paths: Optional[List[Union[str, Path]]] = None,
        use_calibration: bool = True,
        calibration_path: Optional[Union[str, Path]] = None,
        ood_threshold: float = 0.5,
    ):
        """
        Initialize the classifier.

        Args:
            model_path: Path to single model checkpoint
            ensemble_paths: List of model paths for ensemble
            use_calibration: Whether to use temperature scaling
            calibration_path: Path to saved calibration temperature
            ood_threshold: Threshold for OOD detection (energy score)
        """
        self.device = DEVICE
        self.use_ensemble = ensemble_paths is not None and len(ensemble_paths) > 0
        self.ensemble_paths = ensemble_paths or []
        self.ood_threshold = ood_threshold

        # Model load status tracking (for readiness checks)
        self.load_status: ModelLoadStatus = ModelLoadStatus(
            model_name="universal_classifier",
            checkpoint_path=str(model_path or BEST_MODEL_PATH),
        )

        # Load models — fail loudly if required checkpoint is missing/corrupt
        if self.use_ensemble:
            self.models = self._load_ensemble(ensemble_paths)
            print(f"Loaded ensemble of {len(self.models)} models")
        else:
            self.model = self._load_single_model(model_path)
            print("Loaded single model")

        self.load_status.loaded = True

        # Calibration
        self.use_calibration = use_calibration
        if use_calibration and calibration_path and Path(calibration_path).exists():
            self._load_calibration(calibration_path)
            print(f"Loaded calibration from {calibration_path}")

        # OOD Detectors
        self._init_ood_detectors()

        # Set all models to eval
        if self.use_ensemble:
            for m in self.models:
                m.eval()
        else:
            self.model.eval()

    def _load_single_model(self, model_path):
        """Load a single model checkpoint with safe loading."""
        if model_path is None:
            model_path = BEST_MODEL_PATH

        model = build_model().to(self.device)

        checkpoint = safe_load_checkpoint(
            checkpoint_path=model_path,
            model=model,
            model_name="universal_classifier",
            strict=True,
        )

        return model

    def _load_ensemble(self, ensemble_paths):
        """Load ensemble of models with safe loading."""
        models = []
        for i, path in enumerate(ensemble_paths):
            model = build_model().to(self.device)

            checkpoint = safe_load_checkpoint(
                checkpoint_path=path,
                model=model,
                model_name=f"universal_classifier_ensemble_{i}",
                strict=True,
            )

            models.append(model)

        return models

    def _load_calibration(self, calibration_path):
        """Load temperature scaling calibration."""
        checkpoint = torch.load(calibration_path, map_location=self.device)
        temperature = checkpoint.get("temperature", 1.0)

        if self.use_ensemble:
            self.models = [
                ModelWithTemperature(m, temperature).to(self.device)
                for m in self.models
            ]
        else:
            self.model = ModelWithTemperature(self.model, temperature).to(self.device)

    def _init_ood_detectors(self):
        """Initialize OOD detectors."""
        if self.use_ensemble:
            # Use first model for OOD detection
            base_model = self.models[0]
        else:
            base_model = self.model

        # Energy-based OOD
        self.energy_ood = EnergyBasedOOD(base_model, self.device)

        # MSP-based OOD
        self.msp_ood = MSPBasedOOD(base_model, self.device)

        # Entropy-based OOD
        self.entropy_ood = EntropyBasedOOD(base_model, self.device)

        # Combined detector
        self.combined_ood = CombinedOODDetector(base_model, self.device)

    def _preprocess(self, image: Union[str, Path, Image.Image, np.ndarray]) -> torch.Tensor:
        """Preprocess image for inference."""
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")

        return test_transform(image).unsqueeze(0).to(self.device)

    def _get_logits_single(self, model, image_tensor):
        """Get logits from a single model."""
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                logits = model(image_tensor)
        return logits

    def _get_logits_ensemble(self, image_tensor):
        """Get averaged logits from ensemble."""
        all_logits = []

        for model in self.models:
            logits = self._get_logits_single(model, image_tensor)
            all_logits.append(logits)

        # Average logits
        avg_logits = torch.stack(all_logits).mean(dim=0)
        return avg_logits

    def predict(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        return_details: bool = False,
    ) -> Dict:
        """
        Predict image class.

        Args:
            image: Input image (path, PIL Image, or numpy array)
            return_details: Whether to return detailed outputs

        Returns:
            Dict with prediction results
        """
        image_tensor = self._preprocess(image)

        # Get logits
        if self.use_ensemble:
            logits = self._get_logits_ensemble(image_tensor)
        else:
            logits = self._get_logits_single(self.model, image_tensor)

        # Probabilities
        probs = F.softmax(logits, dim=1)

        # Predictions
        confidence, pred_idx = torch.max(probs, dim=1)
        pred_idx = pred_idx.item()
        confidence = confidence.item() * 100

        # OOD Detection
        energy_score = self.energy_ood.compute_scores(image_tensor)[0]
        msp_score = self.msp_ood.compute_scores(image_tensor)[0]
        entropy_score = self.entropy_ood.compute_scores(image_tensor)[0]
        combined_score = self.combined_ood.compute_combined_score(
            image_tensor,
            thresholds={
                'energy': OOD_ENERGY_THRESHOLD,
                'msp': OOD_MSP_THRESHOLD,
                'entropy': OOD_ENTROPY_THRESHOLD,
            },
        )[0]

        # Determine if OOD using OR of all three detectors.
        # Each score: higher = more likely OOD.
        # - Energy: lower (more negative) = ID, higher = OOD
        # - MSP: 1 - max_softmax, higher = more OOD
        # - Entropy: higher = more uncertain/OOD
        is_energy_ood = energy_score > OOD_ENERGY_THRESHOLD
        is_msp_ood = msp_score > OOD_MSP_THRESHOLD
        is_entropy_ood = entropy_score > OOD_ENTROPY_THRESHOLD
        is_ood = is_energy_ood or is_msp_ood or is_entropy_ood

        # Top-k predictions
        top_probs, top_indices = torch.topk(probs, k=min(3, NUM_CLASSES))
        top_predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            top_predictions.append({
                "class": CLASS_NAMES[idx.item()],
                "confidence": round(prob.item() * 100, 2),
            })

        result = {
            "class": CLASS_NAMES[pred_idx],
            "confidence": round(confidence, 2),
            "top_predictions": top_predictions,
            "is_ood": bool(is_ood),
            "ood_scores": {
                "energy": round(float(energy_score), 4),
                "msp": round(float(msp_score), 4),
                "entropy": round(float(entropy_score), 4),
                "combined": round(float(combined_score), 4),
            },
        }

        if return_details:
            result["logits"] = logits.cpu().numpy().tolist()
            result["probabilities"] = probs.cpu().numpy().tolist()

        return result

    def predict_batch(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
    ) -> List[Dict]:
        """
        Predict batch of images.

        Args:
            images: List of input images

        Returns:
            List of prediction results
        """
        results = []
        for img in images:
            results.append(self.predict(img))
        return results

    def get_ood_scores(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
    ) -> Dict:
        """Get detailed OOD scores."""
        image_tensor = self._preprocess(image)

        return {
            "energy": float(self.energy_ood.compute_scores(image_tensor)[0]),
            "msp": float(self.msp_ood.compute_scores(image_tensor)[0]),
            "entropy": float(self.entropy_ood.compute_scores(image_tensor)[0]),
            "combined": float(self.combined_ood.compute_combined_score(image_tensor)[0]),
        }


# ==========================================================
# Singleton Instance (for backward compatibility)
# ==========================================================

classifier = UniversalClassifierInference()


# ==========================================================
# Convenience Functions
# ==========================================================

def predict_image(image_path: str) -> Dict:
    """Convenience function for single image prediction."""
    return classifier.predict(image_path)


def predict_batch(image_paths: List[str]) -> List[Dict]:
    """Convenience function for batch prediction."""
    return classifier.predict_batch(image_paths)


def get_ood_scores(image_path: str) -> Dict:
    """Convenience function for OOD scores."""
    return classifier.get_ood_scores(image_path)


# ==========================================================
# CLI Test
# ==========================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("Enter image path: ")

    result = predict_image(image_path)

    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)
    print(f"Class       : {result['class']}")
    print(f"Confidence  : {result['confidence']:.2f}%")
    print(f"Is OOD      : {result['is_ood']}")
    print(f"OOD Scores  : {result['ood_scores']}")
    print("\nTop Predictions:")
    for pred in result['top_predictions']:
        print(f"  {pred['class']}: {pred['confidence']:.2f}%")
    print("=" * 60)