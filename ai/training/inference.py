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

import structlog

logger = structlog.get_logger(__name__)

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

    if Path(model_path).exists():
        checkpoint = safe_load_checkpoint(
            checkpoint_path=model_path,
            model=model,
            model_name="skin_disease_classifier",
            strict=True,
        )
    else:
        logger.warning(f"Skin disease checkpoint file {model_path} not found. Using base model for inference.")

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
            logger.info("Loaded calibrated model", temperature=f"{temperature:.4f}")
        except Exception as e:
            logger.warning(
                "Could not load calibration checkpoint; proceeding with uncalibrated model",
                error=str(e),
            )
    else:
        logger.info("No calibration found; using uncalibrated model")

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
        try:
            self._load_models(binary_model_path, multiclass_model_path)
            self._init_ood_detectors()
            self._init_gradcam()
        except Exception as e:
            logger.warning("Skin Disease Inference model failed to load", error=str(e))

        logger.info(
            "Skin Disease Inference initialized",
            two_stage=self.use_two_stage,
            calibration=self.use_calibration,
            ensemble=self.use_ensemble,
            ensemble_models=len(self.ensemble_paths),
            classes=len(self.class_names),
        )

    def _load_models(self, binary_path, multiclass_path):
        """Load binary and multiclass models."""

        # Binary model (Stage 1)
        if self.use_two_stage:
            if binary_path is None:
                binary_path = CHECKPOINT_DIR / "best_binary_model.pth"

            if Path(binary_path).exists():
                self.binary_model = load_calibrated_model(binary_path) if self.use_calibration else load_model(binary_path, num_classes=2)
                logger.info("Loaded binary model", path=str(binary_path))
            else:
                logger.info(
                    "Binary model not found; falling back to multiclass only",
                    path=str(binary_path),
                )
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
            logger.info("Loaded ensemble models", count=len(self.multiclass_models))
        else:
            if multiclass_path is None:
                multiclass_path = BEST_MODEL_PATH

            self.multiclass_model = load_calibrated_model(multiclass_path) if self.use_calibration else load_model(multiclass_path)
            logger.info("Loaded multiclass model", path=str(multiclass_path))

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
            logger.info("Grad-CAM initialized", layer=str(target_layer))
        else:
            self.gradcam = None
            logger.warning("Could not find target layer for Grad-CAM", target_layer=GRADCAM_TARGET_LAYER)

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
        return_gradcam: bool = False,
        top_k: int = 3,
        prediction_id: Optional[str] = None,
        image_override: Optional[Image.Image] = None,
    ) -> Dict:
        """
        Predict skin disease from image.

        Grad-CAM is intentionally NOT generated inside this method by default.
        It is now an opt-in, separately scheduled step (see
        :meth:`compute_gradcam_for`) so it does not block the request critical
        path. Callers that want the Grad-CAM image inline can pass
        ``return_gradcam=True``, but the recommended pattern is:

        1. Call ``predict(...)`` with ``return_gradcam=False`` and a
           ``prediction_id`` — the response is returned immediately and
           ``gradcam_image`` is ``None``.
        2. Schedule ``compute_gradcam_for(...)`` as a background task using
           the same ``prediction_id`` and the predicted class/confidence
           from the response.

        Args:
            image: Input image (path, PIL Image, or numpy array). Used
                unless ``image_override`` is provided.
            return_gradcam: Whether to generate Grad-CAM inline (off the
                critical path is the default).
            top_k: Number of top predictions to return
            prediction_id: Optional identifier used to derive the
                Grad-CAM filename when ``return_gradcam=True``.
            image_override: Optional pre-decoded ``PIL.Image.Image`` that
                bypasses the disk decode. When ``None`` (default), the
                decoder uses ``image`` as before. The FastAPI request
                handler uses this to share a single PIL decode between
                the universal classifier and the skin classifier. The
                override must be a PIL Image; passing a numpy array or
                a path here is not supported.

        Returns:
            Dict with prediction results
        """
        # When the caller provides a pre-decoded PIL image, skip the
        # disk decode inside ``_preprocess`` by passing the override
        # through. ``_preprocess`` already accepts PIL images directly
        # and applies the same ``.convert("RGB")`` normalisation, so
        # this is a transparent no-op for callers that don't set the
        # override.
        decode_target: Union[str, Path, Image.Image, np.ndarray] = (
            image_override if image_override is not None else image
        )
        image_tensor, original_np = self._preprocess(decode_target)

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

        # OOD Detection — all derived from the pre-computed multiclass
        # logits/probs. The OOD detectors' original implementation ran
        # their own forward pass through ``self.multiclass_model`` (the
        # calibrated wrapper) on the same image. We now reuse the logits
        # already produced by the multiclass forward pass above, which
        # preserves the calibration step (the wrapper is invoked exactly
        # once via ``_get_multiclass_logits``) and avoids 4 redundant
        # forward passes per request.
        #
        # We upcast to fp32 here for the same reason as the universal
        # classifier: on GPU the multiclass model is wrapped in
        # ``ModelWithTemperature`` and run under autocast, but the
        # original OOD detectors ran their own fp32 forward passes. The
        # upcast keeps the numerical behaviour equivalent.
        ood_logits = logits.float()
        ood_probs = probs.float()

        energy_score = self.energy_ood.compute_from_logits(ood_logits)[0]
        msp_score = self.msp_ood.compute_from_probs(ood_probs)[0]
        entropy_score = self.entropy_ood.compute_from_probs(ood_probs)[0]
        combined_score = self.combined_ood.combine_scores(
            energy=np.array([energy_score]),
            msp=np.array([msp_score]),
            entropy=np.array([entropy_score]),
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
        #
        # Grad-CAM is now generated out-of-band by ``compute_gradcam_for`` and
        # is therefore not on the request critical path. The block below is
        # only entered if a caller explicitly opts in with
        # ``return_gradcam=True``; in that case we delegate to the same
        # helper so behaviour is identical to the previous implementation.
        gradcam_image = None
        if return_gradcam and self.gradcam is not None and not is_healthy_pred:
            gradcam_image = self.compute_gradcam_for(
                image=image_tensor,
                original=original_np,
                pred_idx=pred_idx,
                confidence=confidence,
                prediction_id=prediction_id,
            )

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
            "pred_idx": pred_idx,
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

    def compute_gradcam_for(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor],
        pred_idx: int,
        confidence: float,
        prediction_id: Optional[str] = None,
        original: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """
        Generate the Grad-CAM heatmap for an already-classified image.

        This is the same work the Grad-CAM block used to do inline in
        :meth:`predict`. It is now a separately callable method so the
        FastAPI request handler can schedule it as a ``BackgroundTasks``
        task and return the prediction to the client immediately.

        The model performs one extra forward + backward pass on
        ``image_tensor`` and writes a JPEG to ``GRADCAM_DIR``. The returned
        string is the public URL (e.g. ``/results/gradcam_<id>.jpg``) or
        ``None`` on failure.

        Args:
            image: A preprocessed image tensor (preferred, to avoid a
                second decode) OR a path/PIL image/numpy array.
            pred_idx: Predicted class index from the original ``predict`` call.
            confidence: Predicted confidence (used in the legacy filename).
            prediction_id: Optional stable identifier — when provided, the
                file is named ``gradcam_<id>.jpg`` so the frontend can poll
                for it deterministically. When absent, the legacy
                ``gradcam_<pred_idx>_<conf>.jpg`` name is used.
            original: Optional pre-decoded numpy array (RGB) of the
                original image. When provided, it is used directly instead
                of re-decoding the image. If ``image`` is a tensor, you
                should also pass ``original``; otherwise the file will be
                loaded from disk.

        Returns:
            The public URL of the saved Grad-CAM image, or ``None`` on
            failure.
        """
        if self.gradcam is None:
            return None

        try:
            # Lazily import to keep the module-level import surface small
            # and to ensure the directory check is fresh on each call.
            from ai.config import GRADCAM_DIR
            GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

            if isinstance(image, torch.Tensor):
                image_tensor = image
            else:
                image_tensor, _ = self._preprocess(image)

            if original is None:
                # Re-decode the image from disk/path for the overlay. This
                # path is only used when the caller did not pass a tensor
                # plus the matching numpy original.
                if isinstance(image, (str, Path)):
                    original = np.array(Image.open(image).convert("RGB"))
                elif isinstance(image, Image.Image):
                    original = np.array(image.convert("RGB"))
                else:
                    # Last resort: do not generate without an original.
                    return None

            cam = self.gradcam.generate(image_tensor, pred_idx)
            overlay = overlay_heatmap(original, cam)

            if prediction_id:
                gradcam_filename = f"gradcam_{prediction_id}.jpg"
            else:
                gradcam_filename = f"gradcam_{pred_idx}_{confidence:.1f}.jpg"
            gradcam_path = GRADCAM_DIR / gradcam_filename

            cv2.imwrite(
                str(gradcam_path),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
            )
            return f"/results/{gradcam_filename}"
        except Exception as e:
            logger.warning(
                "Grad-CAM generation failed",
                prediction_id=prediction_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            return None

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


def predict_image(
    image_path,
    top_k=3,
    return_gradcam=False,
    prediction_id=None,
    image_override=None,
):
    """Convenience function for backward compatibility."""
    inference = get_inference()
    return inference.predict(
        image_path,
        top_k=top_k,
        return_gradcam=return_gradcam,
        prediction_id=prediction_id,
        image_override=image_override,
    )


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