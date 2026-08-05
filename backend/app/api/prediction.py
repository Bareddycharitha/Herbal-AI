"""
Universal Prediction API Endpoint

Handles image upload, classification, and routing to appropriate pipeline.
"""

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from backend.app.config import get_settings, Settings
from backend.app.services.universal_classifier import get_classifier
from backend.app.utils.file_validator import file_validator, generate_secure_temp_path
from backend.app.utils.logging import get_logger
from backend.app.exceptions import FileValidationError, ModelError

from ai.recommendation.recommendation_engine import get_recommendation
from ai.recommendation.herb_recommendation_engine import get_herb_recommendation

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"]
)


@router.post("/")
async def predict(
    image: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
):
    """
    Universal Prediction Endpoint

    Workflow:
    Image
      │
      ▼
    Universal Image Classifier
      │
      ├── Skin
      │      │
      │      ▼
      │ Skin Disease Model
      │
      ├── Medicinal
      │      │
      │      ▼
      │ Herb Identification Model
      │
      └── Other
             │
             ▼
      Unsupported Image
    """

    # ======================================================
    # Validate Upload
    # ======================================================

    # Read file content for validation
    content = await image.read()
    await image.seek(0)  # Reset for later use

    try:
        # Comprehensive file validation
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
    # Save Temporary File Securely
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

    try:
        # ==================================================
        # Universal Image Classification
        # ==================================================

        classifier = get_classifier()

        # Run inference in thread pool to avoid blocking event loop
        image_result = await run_in_threadpool(classifier.predict, str(temp_path))

        image_type = image_result["class"]
        confidence = image_result["confidence"]

        logger = get_logger(__name__)
        logger.info(
            "Universal classification complete",
            image_type=image_type,
            confidence=confidence,
            is_ood=image_result.get("is_ood", False),
        )

        # ==================================================
        # Skin Pipeline (Disease Module)
        # ==================================================

        if image_type == "Skin":
            result = await run_in_threadpool(get_recommendation, str(temp_path))
            result["image_type"] = image_type
            result["classifier_confidence"] = confidence
            result["classifier_ood_scores"] = image_result.get("ood_scores", {})
            result["classifier_is_ood"] = image_result.get("is_ood", False)
            return result

        # ==================================================
        # Medicinal Image in Disease Module — Reject
        # ==================================================

        elif image_type == "Medicinal":
            return {
                "success": False,
                "image_type": image_type,
                "classifier_confidence": confidence,
                "classifier_ood_scores": image_result.get("ood_scores", {}),
                "classifier_is_ood": image_result.get("is_ood", False),
                "message": (
                    "Please upload a medicinal herb image in the herb identification module."
                ),
            }

        # ==================================================
        # Other Objects
        # ==================================================

        else:
            return {
                "success": False,
                "image_type": "Other",
                "classifier_confidence": confidence,
                "classifier_ood_scores": image_result.get("ood_scores", {}),
                "classifier_is_ood": image_result.get("is_ood", False),
                "message": (
                    "Unsupported image detected.\n"
                    "Please upload a skin disease image in the disease identification module."
                ),
            }

    except Exception as e:
        logger = get_logger(__name__)
        logger.error("Prediction failed", err=str(e), error_type=type(e).__name__)
        raise ModelError(
            message=f"Prediction failed: {e}",
            details={"image_type": image_type if 'image_type' in locals() else "unknown"},
        )

    finally:
        # Cleanup temp file
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass