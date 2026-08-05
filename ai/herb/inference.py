"""
Herb Identification Inference

Features:
- Binary leaf detector pre-filter
- ArcFace metric learning support
- Ensemble inference
- Temperature scaling calibration
- OOD detection (Energy, MSP, Entropy)
- Configurable confidence thresholds
"""

import torch
import torch.nn.functional as F
import json
from pathlib import Path
from typing import List, Dict, Optional, Union
import numpy as np
from PIL import Image

from .config import (
    DEVICE,
    BEST_MODEL_PATH,
    CLASS_MAPPING_PATH,
    IMAGE_SIZE,
    TOP_K,
    CONFIDENCE_THRESHOLD,
    CALIBRATE_AFTER_TRAINING,
    USE_LEAF_DETECTOR,
    LEAF_DETECTOR_THRESHOLD,
    LEAF_DETECTOR_PATH,
    OOD_ENERGY_THRESHOLD,
    OOD_MSP_THRESHOLD,
    OOD_ENTROPY_THRESHOLD,
    ENSEMBLE_SIZE,
    ENSEMBLE_SEEDS,
)

from .model import build_model
from .transforms import test_transforms
from .leaf_detector import LeafDetectorInference

from ai.training.calibration import ModelWithTemperature
from ai.training.ood_detection import EnergyBasedOOD, MSPBasedOOD, EntropyBasedOOD, CombinedOODDetector


# ==========================================================
# Load Class Mapping
# ==========================================================

def load_class_mapping(class_mapping_path=None):
    """Load class mapping from JSON."""
    if class_mapping_path is None:
        class_mapping_path = CLASS_MAPPING_PATH

    if Path(class_mapping_path).exists():
        with open(class_mapping_path, "r") as f:
            class_mapping = json.load(f)
        return class_mapping
    return None


# ==========================================================
# Herb Inference Class
# ==========================================================

class HerbInference:
    """
    Herb Identification with leaf detection, OOD detection, and calibration.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        ensemble_paths: Optional[List[Union[str, Path]]] = None,
        class_mapping: Optional[Dict] = None,
        use_calibration: bool = CALIBRATE_AFTER_TRAINING,
        calibration_path: Optional[Union[str, Path]] = None,
        use_leaf_detector: Optional[bool] = None,  # None = auto-detect based on checkpoint
        leaf_detector_threshold: float = LEAF_DETECTOR_THRESHOLD,
        ood_threshold: float = 0.5,
    ):
        """
        Initialize herb inference.

        Args:
            model_path: Path to single model checkpoint
            ensemble_paths: List of model paths for ensemble
            class_mapping: Class index to name mapping
            use_calibration: Whether to use temperature scaling
            calibration_path: Path to saved calibration temperature
            use_leaf_detector: Whether to use binary leaf detector.
                              If None (default), auto-enable only if checkpoint exists.
            leaf_detector_threshold: Threshold for leaf detection
            ood_threshold: Threshold for OOD detection
        """
        self.device = DEVICE
        self.use_ensemble = ensemble_paths is not None and len(ensemble_paths) > 0
        self.ensemble_paths = ensemble_paths or []

        # Auto-detect leaf detector: enable only if checkpoint exists
        if use_leaf_detector is None:
            self.use_leaf_detector = Path(LEAF_DETECTOR_PATH).exists()
        else:
            self.use_leaf_detector = use_leaf_detector

        self.leaf_detector_threshold = leaf_detector_threshold
        self.ood_threshold = ood_threshold

        # Load class mapping
        if class_mapping is not None:
            self.class_mapping = class_mapping
        else:
            self.class_mapping = load_class_mapping()

        self.num_classes = len(self.class_mapping) if self.class_mapping else 100

        # Load models
        self._load_models(model_path)

        # Leaf detector - only initialize if enabled and checkpoint exists
        if self.use_leaf_detector:
            if Path(LEAF_DETECTOR_PATH).exists():
                self.leaf_detector = LeafDetectorInference(threshold=self.leaf_detector_threshold)
                print("Leaf detector enabled")
            else:
                # Checkpoint doesn't exist - disable leaf detector
                self.use_leaf_detector = False
                self.leaf_detector = None
                print("Leaf detector disabled (checkpoint not found)")
        else:
            self.leaf_detector = None
            print("Leaf detector disabled")

        # Calibration
        self.use_calibration = use_calibration
        if use_calibration and calibration_path and Path(calibration_path).exists():
            self._load_calibration(calibration_path)

        # OOD Detectors
        self._init_ood_detectors()

        # Set all models to eval
        if self.use_ensemble:
            for m in self.models:
                m.eval()
        else:
            self.model.eval()

        print("Herb Inference initialized")
        print(f"  Ensemble: {self.use_ensemble} ({len(self.ensemble_paths)} models)")
        print(f"  Leaf detector: {self.use_leaf_detector}")
        print(f"  Calibration: {self.use_calibration}")
        print(f"  Classes: {self.num_classes}")

    def _load_models(self, model_path):
        """Load model(s) from checkpoint(s)."""
        if self.use_ensemble:
            self.models = []
            for path in self.ensemble_paths:
                model = build_model(self.num_classes).to(self.device)
                checkpoint = torch.load(path, map_location=self.device)

                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                else:
                    state_dict = checkpoint

                # Handle old checkpoint format (model. -> backbone., model.classifier -> classifier)
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("model."):
                        # Check if it's the classifier
                        if k.startswith("model.classifier."):
                            # Map model.classifier.X -> classifier.X
                            new_k = "classifier." + k[len("model.classifier."):]
                        else:
                            # Map model.XXX -> backbone.XXX
                            new_k = "backbone." + k[6:]
                    elif k.startswith("arcface."):
                        new_k = k
                    elif k.startswith("classifier."):
                        new_k = k
                    else:
                        new_k = k
                    new_state_dict[new_k] = v

                model.load_state_dict(new_state_dict)
                self.models.append(model)
        else:
            if model_path is None:
                model_path = BEST_MODEL_PATH

            self.model = build_model(self.num_classes).to(self.device)

            # Handle missing or incompatible checkpoint
            if Path(model_path).exists():
                try:
                    checkpoint = torch.load(model_path, map_location=self.device)

                    if "model_state_dict" in checkpoint:
                        state_dict = checkpoint["model_state_dict"]
                    else:
                        state_dict = checkpoint

                    # Handle old checkpoint format (model. -> backbone., model.classifier -> classifier)
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        if k.startswith("model."):
                            # Check if it's the classifier
                            if k.startswith("model.classifier."):
                                # Map model.classifier.X -> classifier.X
                                new_k = "classifier." + k[len("model.classifier."):]
                            else:
                                # Map model.XXX -> backbone.XXX
                                new_k = "backbone." + k[6:]
                        elif k.startswith("arcface."):
                            new_k = k
                        elif k.startswith("classifier."):
                            new_k = k
                        else:
                            new_k = k
                        new_state_dict[new_k] = v

                    self.model.load_state_dict(new_state_dict)
                    print(f"Loaded herb model from {model_path}")
                except RuntimeError as e:
                    print(f"Warning: Could not load checkpoint (architecture mismatch): {e}")
                    print("Using randomly initialized weights")
            else:
                print(f"Checkpoint not found at {model_path}, using randomly initialized weights")

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

        print(f"Loaded calibration: T={temperature:.4f}")

    def _init_ood_detectors(self):
        """Initialize OOD detectors."""
        base_model = self.models[0] if self.use_ensemble else self.model

        self.energy_ood = EnergyBasedOOD(base_model, self.device)
        self.msp_ood = MSPBasedOOD(base_model, self.device)
        self.entropy_ood = EntropyBasedOOD(base_model, self.device)
        self.combined_ood = CombinedOODDetector(base_model, self.device)

    def _preprocess(self, image: Union[str, Path, Image.Image, np.ndarray]) -> torch.Tensor:
        """Preprocess image for inference."""
        from PIL import Image

        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")

        return test_transforms(image).unsqueeze(0).to(self.device)

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
        return torch.stack(all_logits).mean(dim=0)

    def predict(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        return_details: bool = False,
    ) -> Dict:
        """
        Predict herb from image.

        Pipeline:
        1. Leaf detection (if enabled)
        2. Herb classification
        3. OOD detection
        4. Confidence thresholding

        Args:
            image: Input image
            return_details: Whether to return detailed outputs

        Returns:
            Dict with prediction results
        """
        image_tensor = self._preprocess(image)

        # Step 1: Leaf Detection
        leaf_result = None
        if self.use_leaf_detector and self.leaf_detector is not None:
            # Use the leaf detector's predict method which handles preprocessing
            from PIL import Image
            if isinstance(image, (str, Path)):
                leaf_result = self.leaf_detector.predict(image)
            else:
                # Save temp and predict
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    if isinstance(image, np.ndarray):
                        Image.fromarray(image).save(tmp.name)
                    else:
                        image.save(tmp.name)
                    leaf_result = self.leaf_detector.predict(tmp.name)
                Path(tmp.name).unlink(missing_ok=True)

            if not leaf_result["is_leaf"]:
                return {
                    "herb": "Non-Leaf",
                    "confidence": round(leaf_result["confidence"] * 100, 2),
                    "top_predictions": [],
                    "is_confident": False,
                    "is_leaf": False,
                    "leaf_confidence": round(leaf_result["leaf_prob"] * 100, 2),
                    "message": "Uploaded image does not appear to be a leaf. Please upload a clear image of a medicinal plant leaf.",
                    "rejected_by": "leaf_detector",
                    "is_ood": False,
                    "ood_scores": {},
                }

        # Step 2: Herb Classification
        if self.use_ensemble:
            logits = self._get_logits_ensemble(image_tensor)
        else:
            logits = self._get_logits_single(self.model, image_tensor)

        # For ArcFace, logits are already scaled cosine similarities
        probs = F.softmax(logits, dim=1)

        confidence, prediction = torch.max(probs, dim=1)
        prediction = prediction.item()
        confidence = confidence.item()

        # Top-K predictions
        top_probs, top_indices = torch.topk(probs, k=min(TOP_K, self.num_classes))

        top_predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            idx = idx.item()
            class_name = self.class_mapping.get(str(idx), f"Class_{idx}")
            top_predictions.append({
                "class": class_name,
                "confidence": round(prob.item() * 100, 2),
            })

        # Step 3: OOD Detection
        energy_score = self.energy_ood.compute_scores(image_tensor)[0]
        msp_score = self.msp_ood.compute_scores(image_tensor)[0]
        entropy_score = self.entropy_ood.compute_scores(image_tensor)[0]
        combined_score = self.combined_ood.compute_combined_score(image_tensor)[0]

        is_ood = (energy_score > OOD_ENERGY_THRESHOLD or
                  msp_score > OOD_MSP_THRESHOLD or
                  entropy_score > OOD_ENTROPY_THRESHOLD)

        # Step 4: Confidence Check
        is_confident = confidence >= CONFIDENCE_THRESHOLD and not is_ood

        herb_name = self.class_mapping.get(str(prediction), f"Class_{prediction}")

        result = {
            "herb": herb_name,
            "confidence": round(confidence * 100, 2),
            "top_predictions": top_predictions,
            "is_confident": is_confident,
            "is_leaf": leaf_result["is_leaf"] if leaf_result else True,
            "leaf_confidence": round(leaf_result["leaf_prob"] * 100, 2) if leaf_result else 100.0,
            "ood_scores": {
                "energy": round(float(energy_score), 4),
                "msp": round(float(msp_score), 4),
                "entropy": round(float(entropy_score), 4),
                "combined": round(float(combined_score), 4),
            },
            "is_ood": bool(is_ood),
        }

        if not is_confident:
            if not leaf_result or not leaf_result["is_leaf"]:
                result["message"] = "Image rejected by leaf detector."
            elif is_ood:
                result["message"] = "Image appears to be out of distribution (not a known herb)."
            else:
                result["message"] = f"Confidence below threshold ({CONFIDENCE_THRESHOLD*100:.0f}%)."

        if return_details:
            result["logits"] = logits.cpu().numpy().tolist()
            result["probabilities"] = probs.cpu().numpy().tolist()

        return result

    def predict_batch(self, images: List[Union[str, Path, Image.Image, np.ndarray]]) -> List[Dict]:
        """Predict batch of images."""
        return [self.predict(img) for img in images]


# ==========================================================
# Singleton Instance (for backward compatibility)
# ==========================================================

_herb_predictor = None


def get_herb_predictor() -> HerbInference:
    """Get or create singleton herb predictor."""
    global _herb_predictor
    if _herb_predictor is None:
        _herb_predictor = HerbInference()
    return _herb_predictor


# Backward compatibility
herb_predictor = get_herb_predictor()


def predict_herb(image_path):
    """Convenience function for backward compatibility."""
    return herb_predictor.predict(image_path)


# ==========================================================
# CLI Test
# ==========================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("Enter image path: ")

    result = predict_herb(image_path)

    print("\n" + "=" * 60)
    print("HERB IDENTIFICATION RESULT")
    print("=" * 60)
    print(f"Herb         : {result['herb']}")
    print(f"Confidence   : {result['confidence']:.2f}%")
    print(f"Is Confident : {result['is_confident']}")
    print(f"Is Leaf      : {result['is_leaf']}")
    print(f"Leaf Conf    : {result['leaf_confidence']:.2f}%")
    print(f"Is OOD       : {result['is_ood']}")
    print(f"OOD Scores   : {result['ood_scores']}")
    print(f"Message      : {result.get('message', 'N/A')}")

    print("\nTop Predictions:")
    for pred in result['top_predictions']:
        print(f"  {pred['class']}: {pred['confidence']:.2f}%")
    print("=" * 60)