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

from pathlib import Path

from ai.training.inference import predict_image, get_inference
from ai.recommendation.knowledge_base import KnowledgeBase
from ai.recommendation.herbal_knowledge_base import HerbalKnowledgeBase
from ai.llm.summary_engine import SummaryEngine

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

import os

# Enable hot-reload in development
HOT_RELOAD = os.getenv("ENVIRONMENT", "development") == "development"

disease_kb = KnowledgeBase(DISEASE_KB_PATH, hot_reload=HOT_RELOAD)

herbal_kb = HerbalKnowledgeBase(HERBAL_KB_PATH, hot_reload=HOT_RELOAD)

summary_engine = SummaryEngine()

image_validator = ImageValidator()
medical_image_validator = MedicalImageValidator()


# ==========================================================
# Recommendation Engine
# ==========================================================


def get_recommendation(image_path):
    """
    Main recommendation pipeline for skin disease analysis.

    Pipeline:
    1. Basic Image Validation
    2. Medical Image Validation (skin-specific)
    3. Skin Disease Prediction (with OOD detection)
    4. Healthy Skin Handling
    5. Disease Information Lookup
    6. Herbal Recommendations
    7. AI Summary Generation
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

    prediction_result = predict_image(image_path)

    print("=" * 60)
    print("PREDICTION RESULT:")
    print(prediction_result)
    print("=" * 60)

    # Extract key information
    prediction = prediction_result["prediction"]
    disease = prediction["disease"]
    confidence = prediction["confidence"]
    confidence_level = prediction["confidence_level"]
    message = prediction_result["message"]
    top_predictions = prediction_result["top_predictions"]
    gradcam_image = prediction_result["gradcam_image"]
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
            "ai_summary": (
                "The uploaded image appears to show healthy skin. "
                "Maintain a healthy skincare routine, moisturize regularly, "
                "use sunscreen daily, stay hydrated, and consult a dermatologist "
                "if you notice any unusual skin changes."
            ),
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
            "ai_summary": (
                "The model detected this image may not be a typical skin lesion. "
                "Please ensure you're uploading a well-lit, close-up photo of the skin condition. "
                "If this is a skin image, the prediction may be unreliable."
            ),
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
            "prediction": prediction,
            "message": message,
            "top_predictions": top_predictions,
            "gradcam_image": gradcam_image,
            "disease_information": None,
            "recommended_herbs": [],
            "herb_details": {},
            "ai_summary": (
                "The model confidence is low for this prediction. "
                "Please upload a clearer, well-lit close-up image of the affected skin."
            ),
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

    disease_information = disease_kb.get_disease_information(disease)

    recommendations = disease_kb.get_recommendations(disease)

    herb_details = {}

    for herb in recommendations:
        details = herbal_kb.get_herb(herb["name"])
        if details:
            herb_details[herb["name"]] = details

    # ======================================================
    # AI Summary Generation
    # ======================================================

    try:
        ai_summary = summary_engine.generate_summary(
            prediction=disease,
            confidence=confidence,
            disease_information=disease_information,
            herbs=recommendations,
        )
    except Exception as e:
        ai_summary = (
            "AI summary could not be generated.\n\n"
            + str(e)
        )

    # ======================================================
    # Final Response
    # ======================================================

    return {
        "success": True,
        "prediction": prediction,
        "message": message,
        "top_predictions": top_predictions,
        "gradcam_image": gradcam_image,
        "disease_information": disease_information,
        "recommended_herbs": recommendations,
        "herb_details": herb_details,
        "ai_summary": ai_summary,
        "binary_stage": binary_stage,
        "ood_scores": ood_scores,
        "is_ood": is_ood,
        "validation_details": {
            "basic": validation,
            "medical": medical_validation,
        },
    }