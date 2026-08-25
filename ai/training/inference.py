"""
Skin Disease Classifier Inference

Supports:
- Two-stage inference: Binary (Healthy/Diseased) -> 22-class Disease
- Ensemble inference
- Temperature scaling calibration
- OOD detection (Energy, MSP, Entropy)
- Grad-CAM explainability
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import json

from ai.config import (
    DEVICE,
    CHECKPOINT_DIR,
    BEST_MODEL_PATH,
    IMAGE_SIZE,
    NUM_CLASSES,
    HIGH_CONFIDENCE_THRESHOLD,
    MEDIUM_CONFIDENCE_THRESHOLD,
    CALIBRATE_AFTER_TRAINING,
    USE_TWO_STAGE,
    OOD_ENERGY_THRESHOLD,
    OOD_MSP_THRESHOLD,
    OOD_ENTROPY_THRESHOLD,
    GRADCAM_TARGET_LAYER,
    USE_GRADCAM_PLUS,
)

import logging

logger = logging.getLogger(__name__)

from ai.models.efficientnet import build_model
from ai.preprocessing.transforms import get_valid_transforms
from ai.preprocessing.dataset import create_dataloaders
from ai.explainability.gradcam_engine import GradCAM, overlay_heatmap
from ai.training.calibration import ModelWithTemperature
from ai.training.ood_detection import EnergyBasedOOD, MSPBasedOOD, EntropyBasedOOD, CombinedOODDetector
from ai.utils.model_loader import safe_load_checkpoint, ModelLoadError

from ai.utils.history import HistoryLogger


# ==========================================================
# Load Class Names (lazy loading)
# ==========================================================

class_names = None  # Will be loaded lazily


def get_class_names():
    """Get class names, loading from dataset if needed."""
    global class_names
    if class_names is None:
        try:
            from ai.config import TRAIN_DIR
            _, _, class_names, _ = create_dataloaders(TRAIN_DIR)
        except Exception:
            # Fallback: default 22 classes
            from ai.config import NUM_CLASSES
            class_names = [f"Class_{i}" for i in range(NUM_CLASSES)]
    return class_names


def load_class_names_from_checkpoint(checkpoint_path):
    """Load class names from checkpoint or dataset."""
    # Try to load from checkpoint metadata
    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        if "class_names" in checkpoint:
            return checkpoint["class_names"]
    except (FileNotFoundError, RuntimeError, Exception) as e:
        logger.warning(
            "Could not load class names from checkpoint",
            path=str(checkpoint_path),
            error=str(e),
        )

    # Fallback: load from dataset
    try:
        from ai.config import TRAIN_DIR
        _, _, class_names, _ = create_dataloaders(TRAIN_DIR)
        return class_names
    except Exception:
        # Last resort: default class names from config
        return [f"Class_{i}" for i in range(NUM_CLASSES)]


# ==========================================================
# Model Loading
# ==========================================================

def load_model(model_path: Union[str, Path], num_classes: int = None):
    """Load model from checkpoint with safe loading."""
    if num_classes is None:
        num_classes = NUM_CLASSES

    model = build_model(num_classes=num_classes).to(DEVICE)

    checkpoint = safe_load_checkpoint(
        checkpoint_path=model_path,
        model=model,
        model_name="skin_disease_classifier",
        strict=True,
    )

    model.eval()
    return model


def load_calibrated_model(model_path: Union[str, Path], cal_path: Union[str, Path] = None):
    """Load model with temperature scaling calibration."""
    model = load_model(model_path)

    if cal_path is None:
        cal_path = Path(model_path).parent / "temperature_scale.pth"

    if Path(cal_path).exists():
        try:
            cal_checkpoint = torch.load(cal_path, map_location=DEVICE, weights_only=False)
            temperature = cal_checkpoint.get("temperature", 1.0)
            model = ModelWithTemperature(model, temperature).to(DEVICE)
            print(f"Loaded calibrated model with T={temperature:.4f}")
        except Exception as e:
            print(f"Warning: Could not load calibration checkpoint: {e}")
            print("Proceeding with uncalibrated model")
    else:
        print("No calibration found, using uncalibrated model")

    return model


# ==========================================================
# Skin Disease Inference Class
# ==========================================================

class SkinDiseaseInference:
    """
    Skin Disease Classifier with two-stage inference, OOD detection, and calibration.
    """

    def __init__(
        self,
        binary_model_path: Optional[Union[str, Path]] = None,
        multiclass_model_path: Optional[Union[str, Path]] = None,
        use_two_stage: bool = USE_TWO_STAGE,
        use_calibration: bool = CALIBRATE_AFTER_TRAINING,
        ensemble_paths: Optional[List[Union[str, Path]]] = None,
        class_names: Optional[List[str]] = None,
    ):
        """
        Initialize skin disease inference.

        Args:
            binary_model_path: Path to binary (Healthy/Diseased) model
            multiclass_model_path: Path to 22-class disease model
            use_two_stage: Whether to use two-stage inference
            use_calibration: Whether to use temperature scaling
            ensemble_paths: List of model paths for ensemble
            class_names: List of disease class names
        """
        self.device = DEVICE
        self.use_two_stage = use_two_stage
        self.use_calibration = use_calibration
        self.use_ensemble = ensemble_paths is not None and len(ensemble_paths) > 0
        self.ensemble_paths = ensemble_paths or []

        # Load class names
        if class_names is not None:
            self.class_names = class_names
        else:
            if multiclass_model_path:
                self.class_names = load_class_names_from_checkpoint(multiclass_model_path)
            else:
                self.class_names = load_class_names_from_checkpoint(BEST_MODEL_PATH)

        # Healthy class index
        self.healthy_idx = self.class_names.index("Unknown_Normal") if "Unknown_Normal" in self.class_names else None

        # Load models
        self._load_models(binary_model_path, multiclass_model_path)

        # OOD Detectors
        self._init_ood_detectors()

        # Grad-CAM
        self._init_gradcam()

        print("Skin Disease Inference initialized")
        print(f"  Two-stage: {self.use_two_stage}")
        print(f"  Calibration: {self.use_calibration}")
        print(f"  Ensemble: {self.use_ensemble} ({len(self.ensemble_paths)} models)")
        print(f"  Classes: {len(self.class_names)}")

    def _load_models(self, binary_path, multiclass_path):
        """Load binary and multiclass models."""

        # Binary model (Stage 1)
        if self.use_two_stage:
            if binary_path is None:
                binary_path = CHECKPOINT_DIR / "best_binary_model.pth"

            if Path(binary_path).exists():
                self.binary_model = load_calibrated_model(binary_path) if self.use_calibration else load_model(binary_path, num_classes=2)
                print(f"Loaded binary model from {binary_path}")
            else:
                print(f"Binary model not found at {binary_path}, will use multiclass only")
                self.use_two_stage = False
                self.binary_model = None
        else:
            self.binary_model = None

        # Multiclass model (Stage 2 or single-stage)
        if self.use_ensemble:
            self.multiclass_models = []
            for path in self.ensemble_paths:
                model = load_calibrated_model(path) if self.use_calibration else load_model(path)
                self.multiclass_models.append(model)
            print(f"Loaded {len(self.multiclass_models)} ensemble models")
        else:
            if multiclass_path is None:
                multiclass_path = BEST_MODEL_PATH

            self.multiclass_model = load_calibrated_model(multiclass_path) if self.use_calibration else load_model(multiclass_path)
            print(f"Loaded multiclass model from {multiclass_path}")

    def _init_ood_detectors(self):
        """Initialize OOD detectors on the main multiclass model."""
        base_model = self.multiclass_models[0] if self.use_ensemble else self.multiclass_model

        self.energy_ood = EnergyBasedOOD(base_model, self.device)
        self.msp_ood = MSPBasedOOD(base_model, self.device)
        self.entropy_ood = EntropyBasedOOD(base_model, self.device)
        self.combined_ood = CombinedOODDetector(base_model, self.device)

    def _init_gradcam(self):
        """Initialize Grad-CAM for explainability."""
        base_model = self.multiclass_models[0] if self.use_ensemble else self.multiclass_model

        # Find target layer
        target_layer = None
        for name, module in base_model.named_modules():
            if GRADCAM_TARGET_LAYER in name:
                target_layer = module
                break

        if target_layer is None:
            # Fallback: try common layer names
            for name in ['conv_head', 'features', 'blocks']:
                for n, m in base_model.named_modules():
                    if name in n and isinstance(m, nn.Conv2d):
                        target_layer = m
                        break
                if target_layer:
                    break

        if target_layer:
            self.gradcam = GradCAM(base_model, target_layer)
            print(f"Grad-CAM initialized on layer: {target_layer}")
        else:
            self.gradcam = None
            print("Warning: Could not find target layer for Grad-CAM")

    def _preprocess(self, image: Union[str, Path, Image.Image, np.ndarray]) -> Tuple[torch.Tensor, np.ndarray]:
        """Preprocess image for inference. Returns (tensor, original_numpy)."""
        if isinstance(image, (str, Path)):
            original = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            original = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            original = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        original_np = np.array(original)
        transform = get_valid_transforms()
        tensor = transform(image=original_np)["image"]
        tensor = tensor.unsqueeze(0).to(self.device)

        return tensor, original_np

    def _get_multiclass_logits(self, image_tensor):
        """Get logits from multiclass model(s)."""
        if self.use_ensemble:
            all_logits = []
            for model in self.multiclass_models:
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                        logits = model(image_tensor)
                all_logits.append(logits)
            return torch.stack(all_logits).mean(dim=0)
        else:
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                    logits = self.multiclass_model(image_tensor)
            return logits

    def _get_binary_logits(self, image_tensor):
        """Get logits from binary model."""
        if self.binary_model is None:
            return None

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                logits = self.binary_model(image_tensor)
        return logits

    def predict(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        return_gradcam: bool = True,
        top_k: int = 3,
    ) -> Dict:
        """
        Predict skin disease from image.

        Args:
            image: Input image
            return_gradcam: Whether to generate Grad-CAM
            top_k: Number of top predictions to return

        Returns:
            Dict with prediction results
        """
        image_tensor, original_np = self._preprocess(image)

        # Stage 1: Binary classification (if enabled)
        is_healthy = False
        healthy_confidence = 0.0

        if self.use_two_stage and self.binary_model is not None:
            binary_logits = self._get_binary_logits(image_tensor)
            binary_probs = F.softmax(binary_logits, dim=1)
            healthy_prob = binary_probs[0, 0].item()
            diseased_prob = binary_probs[0, 1].item()

            is_healthy = healthy_prob > diseased_prob
            healthy_confidence = max(healthy_prob, diseased_prob) * 100

            if is_healthy:
                # Healthy skin - return early
                return self._format_healthy_result(healthy_confidence, original_np)

        # Stage 2: Multi-class disease classification
        logits = self._get_multiclass_logits(image_tensor)
        probs = F.softmax(logits, dim=1)

        # Top-k predictions
        top_probs, top_indices = torch.topk(probs, k=min(top_k, len(self.class_names)))

        top_predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            idx = idx.item()
            top_predictions.append({
                "disease": self.class_names[idx],
                "confidence": round(prob.item() * 100, 2),
            })

        # Primary prediction
        pred_idx = top_indices[0, 0].item()
        confidence = top_predictions[0]["confidence"]
        disease = top_predictions[0]["disease"]

        # Check if predicted as healthy
        is_healthy_pred = (self.healthy_idx is not None and pred_idx == self.healthy_idx)

        # Confidence level
        if confidence >= HIGH_CONFIDENCE_THRESHOLD:
            confidence_level = "high"
            message = "Prediction confidence is high. Recommendations can be used as educational guidance."
        elif confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
            confidence_level = "medium"
            message = "Prediction confidence is moderate. Please verify the condition before relying on recommendations."
        else:
            confidence_level = "low"
            message = "Prediction confidence is low. Herbal recommendations will not be provided. Please consult a dermatologist."

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

        # OOD detection using OR of all three detectors
        # Each score: higher = more likely OOD
        is_energy_ood = energy_score > OOD_ENERGY_THRESHOLD
        is_msp_ood = msp_score > OOD_MSP_THRESHOLD
        is_entropy_ood = entropy_score > OOD_ENTROPY_THRESHOLD
        is_ood = is_energy_ood or is_msp_ood or is_entropy_ood

        # Grad-CAM
        gradcam_image = None
        if return_gradcam and self.gradcam is not None and not is_healthy_pred:
            try:
                cam = self.gradcam.generate(image_tensor, pred_idx)
                overlay = overlay_heatmap(original_np, cam)

                # Save Grad-CAM
                from ai.config import RESULTS_DIR
                RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                gradcam_filename = f"gradcam_{pred_idx}_{confidence:.1f}.jpg"
                gradcam_path = RESULTS_DIR / gradcam_filename
                cv2.imwrite(str(gradcam_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                gradcam_image = f"/results/{gradcam_filename}"
            except Exception as e:
                print(f"Grad-CAM generation failed: {e}")

        return {
            "success": True,
            "prediction": {
                "disease": "Healthy Skin" if is_healthy_pred else disease,
                "confidence": confidence,
                "confidence_level": confidence_level,
            },
            "message": message,
            "top_predictions": top_predictions,
            "gradcam_image": gradcam_image,
            "is_healthy": is_healthy_pred,
            "binary_stage": {
                "used": self.use_two_stage and self.binary_model is not None,
                "is_healthy": is_healthy,
                "confidence": healthy_confidence,
            } if self.use_two_stage else None,
            "ood_scores": {
                "energy": round(float(energy_score), 4),
                "msp": round(float(msp_score), 4),
                "entropy": round(float(entropy_score), 4),
                "combined": round(float(combined_score), 4),
            },
            "is_ood": bool(is_ood),
        }

    def _format_healthy_result(self, confidence, original_np):
        """Format result for healthy skin prediction."""
        return {
            "success": True,
            "prediction": {
                "disease": "Healthy Skin",
                "confidence": round(confidence, 2),
                "confidence_level": "high" if confidence >= 70 else "medium",
            },
            "message": "No visible skin disease was detected.",
            "top_predictions": [{"disease": "Healthy Skin", "confidence": round(confidence, 2)}],
            "gradcam_image": None,
            "is_healthy": True,
            "binary_stage": {
                "used": True,
                "is_healthy": True,
                "confidence": round(confidence, 2),
            },
            "ood_scores": {},
            "is_ood": False,
        }

    def predict_batch(self, images: List[Union[str, Path, Image.Image, np.ndarray]]) -> List[Dict]:
        """Predict batch of images."""
        return [self.predict(img) for img in images]


# ==========================================================
# Singleton Instance (for backward compatibility)
# ==========================================================

_inference_instance = None


def get_inference() -> SkinDiseaseInference:
    """Get or create singleton inference instance."""
    global _inference_instance
    if _inference_instance is None:
        _inference_instance = SkinDiseaseInference()
    return _inference_instance


def predict_image(image_path, top_k=3):
    """Convenience function for backward compatibility."""
    inference = get_inference()
    return inference.predict(image_path, top_k=top_k)


# ==========================================================
# CLI Test
# ==========================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("Enter image path: ")

    inference = SkinDiseaseInference()
    result = inference.predict(image_path)

    print("\n" + "=" * 60)
    print("SKIN DISEASE PREDICTION")
    print("=" * 60)
    print(f"Disease      : {result['prediction']['disease']}")
    print(f"Confidence   : {result['prediction']['confidence']:.2f}%")
    print(f"Conf Level   : {result['prediction']['confidence_level']}")
    print(f"Is Healthy   : {result['is_healthy']}")
    print(f"Is OOD       : {result['is_ood']}")
    print(f"OOD Scores   : {result['ood_scores']}")
    print(f"Message      : {result['message']}")
    print(f"Grad-CAM     : {result['gradcam_image']}")

    if result['binary_stage']:
        print(f"\nBinary Stage : Healthy={result['binary_stage']['is_healthy']}, Conf={result['binary_stage']['confidence']:.2f}%")

    print("\nTop Predictions:")
    for pred in result['top_predictions']:
        print(f"  {pred['disease']}: {pred['confidence']:.2f}%")
    print("=" * 60)