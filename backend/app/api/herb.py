"""
Herb Identification API Endpoint

Direct herb identification from uploaded leaf images.
"""

import shutil
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from backend.app.config import get_settings, Settings
from backend.app.services.universal_classifier import get_classifier
from backend.app.utils.file_validator import file_validator, generate_secure_temp_path
from backend.app.utils.logging import get_logger
from backend.app.exceptions import FileValidationError, ModelError, ModelLoadError

from ai.recommendation.herb_recommendation_engine import get_herb_recommendation

router = APIRouter(
    prefix="/herb",
    tags=["Herb Identification"],
)


@router.post("/")
async def predict_herb(
    image: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
):
    """
    Identify a medicinal herb from an uploaded image.
    """

    # ======================================================
    # Validate Image
    # ======================================================

    content = await image.read()
    await image.seek(0)

    try:
        detected_mime = file_validator.validate(
            content=content,
            filename=image.filename or "unknown",
            declared_mime=image.content_type,
        )
    except FileValidationError as e:
        raise HTTPException(
            status_code=e.status_code,
            detail=e.to_dict(),
        )

    # ======================================================
    # Save Temporary Image
    # ======================================================

    suffix = Path(image.filename).suffix.lower() if image.filename else ".jpg"
    temp_path = generate_secure_temp_path(suffix=suffix)

    try:
        with open(temp_path, "wb") as temp_file:
            temp_file.write(content)
    except Exception as e:
        raise ModelError(
            message=f"Failed to save uploaded file: {e}",
            details={"filename": image.filename},
        )

    # ======================================================
    # Universal Image Classification
    # ======================================================

    try:
        classifier = get_classifier()
        class_result = await run_in_threadpool(classifier.predict, str(temp_path))
        image_type = class_result["class"]
        classifier_confidence = class_result["confidence"]
        classifier_ood_scores = class_result.get("ood_scores", {})
        classifier_is_ood = class_result.get("is_ood", False)
    except AttributeError as e:
        # The universal classifier swallows checkpoint-load failures; the
        # AttributeError surfaces here when it tries to call .eval() on a
        # None model. Map it to a structured 503 so the browser can show
        # a real message instead of a generic "Network Error".
        msg = str(e)
        if "NoneType" in msg and "eval" in msg:
            logger = get_logger(__name__)
            logger.error(
                "Universal classifier model not loaded (herb endpoint)",
                err=msg,
                hint=(
                    "Place a trained checkpoint at "
                    "ai/image_classifier/checkpoints/best_model.pth."
                ),
            )
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise ModelLoadError(
                model_path="ai/image_classifier/checkpoints/best_model.pth",
                reason="Universal classifier checkpoint not loaded",
            )
        raise
    except FileNotFoundError as e:
        logger = get_logger(__name__)
        logger.error(
            "Model checkpoint missing in herb endpoint",
            err=str(e),
            missing_path=str(e.filename) if getattr(e, "filename", None) else None,
        )
        raise ModelLoadError(
            model_path=str(getattr(e, "filename", "ai/checkpoints/")),
            reason=str(e),
        )
    except Exception as e:
        logger = get_logger(__name__)
        logger.error("Universal classification failed in herb endpoint", err=str(e), error_type=type(e).__name__)
        raise ModelError(
            message=f"Image classification failed: {e}",
        )

    has_checkpoint = getattr(classifier, "checkpoint_found", True)

    # Reject skin/disease images in the herb module if checkpoint exists
    if image_type == "Skin" and has_checkpoint:
        return {
            "success": False,
            "image_type": image_type,
            "classifier_confidence": classifier_confidence,
            "classifier_ood_scores": classifier_ood_scores,
            "classifier_is_ood": classifier_is_ood,
            "message": (
                "Please upload a skin disease image in the disease identification module."
            ),
        }

    # Reject unsupported images in the herb module if checkpoint exists
    if image_type == "Other" and has_checkpoint:
        return {
            "success": False,
            "image_type": image_type,
            "classifier_confidence": classifier_confidence,
            "classifier_ood_scores": classifier_ood_scores,
            "classifier_is_ood": classifier_is_ood,
            "message": (
                "Unsupported image detected.\n"
                "Please upload a medicinal herb leaf image in the herb identification module."
            ),
        }

    # ======================================================
    # Prediction (using recommendation engine)
    # ======================================================

    try:
        # Run inference in thread pool
        result = await run_in_threadpool(get_herb_recommendation, str(temp_path))

        result["image_type"] = image_type
        result["classifier_confidence"] = classifier_confidence
        result["classifier_ood_scores"] = classifier_ood_scores
        result["classifier_is_ood"] = classifier_is_ood

        return result

    except FileNotFoundError as e:
        logger = get_logger(__name__)
        logger.error(
            "Herb checkpoint missing",
            err=str(e),
            missing_path=str(e.filename) if getattr(e, "filename", None) else None,
        )
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise ModelLoadError(
            model_path=str(getattr(e, "filename", "ai/herb/checkpoints/")),
            reason=str(e),
        )
    except Exception as e:
        logger = get_logger(__name__)
        logger.error("Herb prediction failed", err=str(e), error_type=type(e).__name__)
        raise ModelError(
            message=f"Herb identification failed: {e}",
        )

    finally:
        # Cleanup temp file
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass