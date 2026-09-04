"""
Herb Recommendation Engine

Pipeline:
1. Image Validation
2. Leaf Detection (pre-filter)
3. Herb Identification with OOD detection
4. Knowledge Base Lookup
5. AI Summary Generation
"""

from ai.herb.inference import get_herb_predictor
from ai.herb.knowledge_base import herb_knowledge
from ai.llm.summary_engine import SummaryEngine
from ai.validation.image_validator import ImageValidator
from ai.validation.medical_image_validator import MedicalImageValidator


# ==========================================================
# Load Components
# ==========================================================

summary_engine = SummaryEngine()
image_validator = ImageValidator()
medical_image_validator = MedicalImageValidator()
herb_predictor = get_herb_predictor()


# ==========================================================
# Herb Recommendation Engine
# ==========================================================


def get_herb_recommendation(image_path):

    # ======================================================
    # Basic Image Validation
    # ======================================================

    validation = image_validator.validate(image_path)

    if not validation["success"]:
        return validation

    # ======================================================
    # Leaf-Specific Validation
    # ======================================================

    # Use medical validator with adjusted thresholds for leaves
    # Leaves have different color profiles than skin
    leaf_validation = medical_image_validator.validate(image_path)

    # For leaf images, we don't strictly require skin detection
    # but we can use edge detection to check for leaf-like structures
    if not leaf_validation["success"]:
        # Check if it's a leaf-like image (high edge ratio might indicate leaf veins)
        # We'll be more lenient for herb images
        pass  # Continue anyway, leaf detector will catch non-leaves

    # ======================================================
    # Herb Prediction (with leaf detection & OOD)
    # ======================================================

    prediction = herb_predictor.predict(image_path)

    herb = prediction["herb"]
    confidence = prediction["confidence"]
    top_predictions = prediction["top_predictions"]
    is_confident = prediction["is_confident"]
    is_leaf = prediction.get("is_leaf", True)
    leaf_confidence = prediction.get("leaf_confidence", 100.0)
    is_ood = prediction.get("is_ood", False)
    ood_scores = prediction.get("ood_scores", {})
    rejected_by = prediction.get("rejected_by", None)

    # ======================================================
    # Handle Rejection by Leaf Detector
    # ======================================================

    if not is_leaf:
        return {
            "success": False,
            "prediction": prediction,
            "message": prediction.get("message", "Uploaded image does not appear to be a medicinal leaf."),
            "top_predictions": top_predictions,
            "herb_information": None,
            "ai_summary": None,
            "rejected_by": "leaf_detector",
            "leaf_confidence": leaf_confidence,
            "ood_scores": ood_scores,
            "is_ood": is_ood,
            "validation_details": {
                "basic": validation,
            },
        }

    # ======================================================
    # Handle OOD Detection
    # ======================================================

    if is_ood:
        return {
            "success": False,
            "prediction": prediction,
            "message": (
                "The image appears to be outside the known herb classes. "
                "Please upload a clear image of a known medicinal plant leaf."
            ),
            "top_predictions": top_predictions,
            "herb_information": None,
            "ai_summary": None,
            "rejected_by": "ood_detector",
            "ood_scores": ood_scores,
            "is_ood": is_ood,
            "validation_details": {
                "basic": validation,
            },
        }

    # ======================================================
    # Get Herb Information
    # ======================================================

    herb_information = herb_knowledge.get_herb(herb)

    # ======================================================
    # Generate AI Summary
    # ======================================================

    try:
        ai_summary = summary_engine.generate_herb_summary(
            herb=herb,
            herb_information=herb_information,
        )
    except Exception as e:
        ai_summary = f"AI summary could not be generated.\n\n{str(e)}"

    # ==========================================================
    # Final Response
    # ==========================================================

    msg = "Medicinal plant identified successfully."
    if not is_confident:
        msg = f"Medicinal plant identified with moderate confidence ({confidence:.1f}%)."

    return {
        "success": True,
        "prediction": prediction,
        "message": msg,
        "top_predictions": top_predictions,
        "herb_information": herb_information,
        "ai_summary": ai_summary,
        "ood_scores": ood_scores,
        "is_ood": is_ood,
        "validation_details": {
            "basic": validation,
        },
    }