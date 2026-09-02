"""
Skin Disease Recommendation Engine

Pipeline:
1. Basic Image Validation
2. Medical Image Validation (skin-specific)
3. Skin Disease Prediction (with OOD detection)
4. Healthy Skin Handling
5. Disease Information Lookup
6. Herbal Recommendations
7. AI Summary Generation
"""

import os
import threading
from pathlib import Path

from ai.training.inference import predict_image
from ai.recommendation.knowledge_base import KnowledgeBase
from ai.recommendation.herbal_knowledge_base import HerbalKnowledgeBase

from ai.validation.image_validator import ImageValidator
from ai.validation.medical_image_validator import MedicalImageValidator


# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DISEASE_KB_PATH = (
    BASE_DIR
    / "datasets"
    / "knowledge_base"
    / "disease_knowledge_base.json"
)

HERBAL_KB_PATH = (
    BASE_DIR
    / "datasets"
    / "knowledge_base"
    / "herbal_knowledge_base.json"
)

# ==========================================================
# Load Components
# ==========================================================

# Hot-reload is enabled in development so editing the JSON knowledge
# bases is reflected without restarting the server. In production the
# files are loaded once on first use.
HOT_RELOAD = os.getenv("ENVIRONMENT", "development") == "development"

# Validators are stateless and cheap to construct; instantiate them
# eagerly so the request path is one indirection lighter.
image_validator = ImageValidator()
medical_image_validator = MedicalImageValidator()

# Knowledge bases are file-backed (each ``__init__`` opens the JSON file
# and may spawn a background watcher thread). They are accessed only
# when a prediction reaches the "Disease Information" / "Herbal
# Recommendations" stage, so we lazy-init them on first access. This
# avoids paying the JSON parse + watcher-thread cost on module import
# (which used to happen even when only the universal classifier was
# needed at startup, e.g. for /health).
#
# ``threading.RLock`` keeps the first-access instantiation safe across
# concurrent worker threads spawned by FastAPI's ``run_in_threadpool``.
_disease_kb_lock = threading.RLock()
_herbal_kb_lock = threading.RLock()
_disease_kb: "KnowledgeBase | None" = None
_herbal_kb: "HerbalKnowledgeBase | None" = None


def _get_disease_kb() -> "KnowledgeBase":
    global _disease_kb
    if _disease_kb is None:
        with _disease_kb_lock:
            if _disease_kb is None:  # double-checked
                _disease_kb = KnowledgeBase(DISEASE_KB_PATH, hot_reload=HOT_RELOAD)
    return _disease_kb


def _get_herbal_kb() -> "HerbalKnowledgeBase":
    global _herbal_kb
    if _herbal_kb is None:
        with _herbal_kb_lock:
            if _herbal_kb is None:  # double-checked
                _herbal_kb = HerbalKnowledgeBase(HERBAL_KB_PATH, hot_reload=HOT_RELOAD)
    return _herbal_kb


# ==========================================================
# Recommendation Engine
# ==========================================================


def get_recommendation(
    image_path,
    prediction_id=None,
    generate_gradcam=False,
    image_override=None,
):
    """
    Main recommendation pipeline for skin disease analysis.

    Grad-CAM is intentionally NOT generated inline here by default — the
    request handler is expected to schedule it as a ``BackgroundTasks``
    task using the returned ``prediction_id`` and the ``pred_idx`` /
    ``confidence`` from the response.

    Args:
        image_path: Path to the uploaded image. Required by the basic
            and medical validators (which use ``cv2.imread``) and is
            also used as the source of truth by the background Grad-CAM
            task. It is **not** used to decode the image for the skin
            model when ``image_override`` is provided.
        prediction_id: Optional stable identifier. When provided, it is
            returned in the response so the caller can correlate the
            background Grad-CAM job with this prediction.
        generate_gradcam: If True, generate the Grad-CAM inline (legacy
            behaviour). Off by default to keep the critical path fast.
        image_override: Optional pre-decoded ``PIL.Image.Image`` that is
            forwarded to the skin model to avoid a duplicate PIL decode
            of the same uploaded file. When ``None`` (default), the
            skin model decodes the image from ``image_path`` as before.

    Pipeline:
    1. Basic Image Validation
    2. Medical Image Validation (skin-specific)
    3. Skin Disease Prediction (with OOD detection)
    4. Healthy Skin Handling
    5. Disease Information Lookup
    6. Herbal Recommendations
    (AI summary is produced separately by /api/v1/summary/, see Priority 1.)
    """

    # ======================================================
    # Basic Image Validation
    # ======================================================

    validation = image_validator.validate(image_path)

    if not validation["success"]:
        return validation

    # ======================================================
    # Medical Image Validation (skin-specific)
    # ======================================================

    medical_validation = medical_image_validator.validate(image_path)

    if not medical_validation["success"]:
        return {
            "success": False,
            "message": medical_validation["message"],
            "validation_details": {
                "basic": validation,
                "medical": medical_validation,
            },
        }

    # ======================================================
    # Disease Prediction (using new inference)
    # ======================================================

    prediction_result = predict_image(
        image_path,
        return_gradcam=generate_gradcam,
        prediction_id=prediction_id,
        image_override=image_override,
    )

    # Extract key information
    prediction = prediction_result["prediction"]
    disease = prediction["disease"]
    confidence = prediction["confidence"]
    confidence_level = prediction["confidence_level"]
    message = prediction_result["message"]
    top_predictions = prediction_result["top_predictions"]
    gradcam_image = prediction_result["gradcam_image"]
    pred_idx = prediction_result.get("pred_idx")
    is_healthy = prediction_result.get("is_healthy", False)
    is_ood = prediction_result.get("is_ood", False)
    binary_stage = prediction_result.get("binary_stage", None)
    ood_scores = prediction_result.get("ood_scores", {})

    # ======================================================
    # Healthy Skin
    # ======================================================

    if is_healthy:
        return {
            "success": True,
            "prediction_id": prediction_id,
            "pred_idx": pred_idx,
            "prediction": {
                "disease": "Healthy Skin",
                "confidence": confidence,
                "confidence_level": confidence_level,
            },
            "message": "No visible skin disease was detected.",
            "top_predictions": top_predictions,
            "gradcam_image": gradcam_image,
            "disease_information": None,
            "recommended_herbs": [],
            "herb_details": {},
            # AI summary is generated separately by POST /api/v1/summary/
            "ai_summary": None,
            "binary_stage": binary_stage,
            "ood_scores": ood_scores,
            "is_ood": is_ood,
            "validation_details": {
                "basic": validation,
                "medical": medical_validation,
            },
        }

    # ======================================================
    # OOD / Uncertain Handling
    # ======================================================

    if is_ood:
        return {
            "success": True,
            "prediction_id": prediction_id,
            "pred_idx": pred_idx,
            "prediction": prediction,
            "message": (
                "The image appears to be outside the expected domain (not a typical skin lesion). "
                "Please upload a clear, close-up image of the affected skin area."
            ),
            "top_predictions": top_predictions,
            "gradcam_image": gradcam_image,
            "disease_information": None,
            "recommended_herbs": [],
            "herb_details": {},
            # AI summary is generated separately by POST /api/v1/summary/
            "ai_summary": None,
            "binary_stage": binary_stage,
            "ood_scores": ood_scores,
            "is_ood": is_ood,
            "validation_details": {
                "basic": validation,
                "medical": medical_validation,
            },
        }

    # ======================================================
    # Low Confidence Handling
    # ======================================================

    if confidence_level == "low":
        return {
            "success": True,
            "prediction_id": prediction_id,
            "pred_idx": pred_idx,
            "prediction": prediction,
            "message": message,
            "top_predictions": top_predictions,
            "gradcam_image": gradcam_image,
            "disease_information": None,
            "recommended_herbs": [],
            "herb_details": {},
            # AI summary is generated separately by POST /api/v1/summary/
            "ai_summary": None,
            "binary_stage": binary_stage,
            "ood_scores": ood_scores,
            "is_ood": is_ood,
            "validation_details": {
                "basic": validation,
                "medical": medical_validation,
            },
        }

    # ======================================================
    # Disease Information
    # ======================================================

    # The two knowledge bases are lazy-initialised here. The first
    # successful skin prediction that reaches this branch pays the
    # JSON parse + watcher-thread cost once; subsequent requests
    # re-use the same instance.
    _disease_kb = _get_disease_kb()
    _herbal_kb = _get_herbal_kb()

    disease_information = _disease_kb.get_disease_information(disease)

    recommendations = _disease_kb.get_recommendations(disease)

    herb_details = {}

    for herb in recommendations:
        details = _herbal_kb.get_herb(herb["name"])
        if details:
            herb_details[herb["name"]] = details

    # ======================================================
    # Final Response
    # ======================================================
    # The AI medical summary is intentionally NOT generated here.
    # It is produced by the dedicated endpoint POST /api/v1/summary/
    # so that the prediction request never has to wait for OpenRouter.
    # See backend/app/api/summary.py for the async summary flow.

    return {
        "success": True,
        "prediction_id": prediction_id,
        "pred_idx": pred_idx,
        "prediction": prediction,
        "message": message,
        "top_predictions": top_predictions,
        "gradcam_image": gradcam_image,
        "disease_information": disease_information,
        "recommended_herbs": recommendations,
        "herb_details": herb_details,
        "ai_summary": None,
        "binary_stage": binary_stage,
        "ood_scores": ood_scores,
        "is_ood": is_ood,
        "validation_details": {
            "basic": validation,
            "medical": medical_validation,
        },
    }