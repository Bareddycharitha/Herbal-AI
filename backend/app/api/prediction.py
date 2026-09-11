"""
Universal Prediction API Endpoint

Handles image upload, classification, and routing to appropriate pipeline.

Grad-CAM is generated as a FastAPI ``BackgroundTasks`` task after the
response has been built. The response is returned to the client immediately
with ``gradcam_image: null`` and a ``prediction_id``; the client can poll
``GET /api/v1/gradcam/{prediction_id}`` to retrieve the public URL once
the heatmap has been written to disk.
"""

import shutil
import tempfile
import uuid
from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, UploadFile, File, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError

from backend.app.api.gradcam import get_job_store
from backend.app.api.history import get_history_service
from backend.app.dependencies import get_optional_user
from backend.app.config import get_settings, Settings
from backend.app.services.universal_classifier import get_classifier
from backend.app.utils.file_validator import file_validator, generate_secure_temp_path
from backend.app.utils.logging import get_logger
from backend.app.exceptions import FileValidationError, ModelError, ModelLoadError

from ai.recommendation.recommendation_engine import get_recommendation
from ai.recommendation.herb_recommendation_engine import get_herb_recommendation

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"]
)


async def _save_history_background(
    profile_id: str,
    result: dict,
    email: str = None,
    name: str = None,
):
    """Save prediction into user's database history."""
    try:
        service = get_history_service()
        await service.record_prediction(
            profile_id=profile_id,
            prediction=result.get("prediction"),
            confidence=result.get("confidence"),
            confidence_level=result.get("confidence_level"),
            top_predictions=result.get("top_predictions"),
            disease_information=result.get("disease_information"),
            recommended_herbs=result.get("herbs"),
            image_path=result.get("gradcam_image"),
            ai_summary=result.get("summary"),
            prediction_id=result.get("prediction_id"),
            user_email=email,
            user_name=name,
        )
    except Exception as e:
        logger = get_logger(__name__)
        logger.error("Failed to save history in background", error=str(e), profile_id=profile_id)


def _run_gradcam_background(
    prediction_id: str,
    image_path: str,
    pred_idx: int,
    confidence: float,
):
    """
    Background-task body: generate the Grad-CAM heatmap for an already-
    classified image and update the in-memory job store with the result.

    This function is intentionally synchronous and CPU-bound; it is run
    inside the FastAPI ``BackgroundTasks`` threadpool which serialises
    them, so it does not block the event loop.
    """
    store = get_job_store()
    try:
        from ai.training.inference import get_inference
        skin = get_inference()
        url = skin.compute_gradcam_for(
            image=image_path,
            pred_idx=pred_idx,
            confidence=confidence,
            prediction_id=prediction_id,
        )
        if url:
            store.mark_ready(prediction_id, url)
        else:
            store.mark_failed(prediction_id, "compute_gradcam_for returned None")
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(
            "Background Grad-CAM failed",
            prediction_id=prediction_id,
            err=str(e),
            error_type=type(e).__name__,
        )
        try:
            store.mark_failed(prediction_id, type(e).__name__)
        except Exception:
            pass
    finally:
        # Clean up the temp file only after the Grad-CAM job is done
        # (success or failure). This is the latest point at which the
        # file is still needed.
        try:
            Path(image_path).unlink(missing_ok=True)
        except Exception:
            pass


@router.post("/")
async def predict(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    current_user = Depends(get_optional_user),
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

    # ======================================================
    # Decode Image Once (shared with both model pipelines)
    # ======================================================
    # Both the universal classifier and the skin disease model need a
    # decoded RGB image. We decode the uploaded bytes ONCE here and
    # forward the resulting PIL image to both, so the JPEG is not
    # re-parsed on disk by each pipeline.
    #
    # The temp file is still required because:
    #   1. The basic and medical validators inside
    #      ``get_recommendation`` use ``cv2.imread(image_path)`` (out of
    #      scope for Priority 3B).
    #   2. The background Grad-CAM task reads the path to compute its
    #      own tensor + overlay.
    try:
        shared_pil_image = Image.open(BytesIO(content)).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        # Best-effort cleanup before bubbling up the error.
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise ModelError(
            message=f"Failed to decode uploaded image: {e}",
            details={"filename": image.filename},
        )

    # Track locals for the except branch.
    image_type: str = "unknown"
    prediction_id: str = str(uuid.uuid4())
    get_job_store().create(prediction_id)

    try:
        # ==================================================
        # Universal Image Classification
        # ==================================================

        classifier = get_classifier()

        # Run inference in thread pool to avoid blocking event loop.
        # We pass the pre-decoded PIL image (Priority 3B) to avoid a
        # second JPEG decode inside ``UniversalClassifierInference._preprocess``.
        image_result = await run_in_threadpool(
            classifier.predict, shared_pil_image
        )

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

        if image_type == "Skin" or not getattr(classifier, "checkpoint_found", True):
            if not getattr(classifier, "checkpoint_found", True) and image_type != "Skin":
                logger.warning(
                    "Universal classifier checkpoint not found (untrained base model). "
                    f"Bypassing domain rejection for predicted class '{image_type}' and running skin pipeline."
                )
            # ``str(temp_path)`` is still passed because the basic and
            # medical validators (inside ``get_recommendation``) use
            # ``cv2.imread(image_path)``. The skin model itself receives
            # the shared PIL image via ``image_override`` to skip a
            # duplicate disk decode (Priority 3B).
            result = await run_in_threadpool(
                get_recommendation,
                str(temp_path),
                prediction_id,
                False,  # generate_gradcam=False: keep Grad-CAM off the critical path
                shared_pil_image,  # image_override
            )
            result["image_type"] = image_type
            result["classifier_confidence"] = confidence
            result["classifier_ood_scores"] = image_result.get("ood_scores", {})
            result["classifier_is_ood"] = image_result.get("is_ood", False)

            # Schedule the Grad-CAM generation as a background task so the
            # response is returned to the client immediately. The job
            # store will be updated when the heatmap is written.
            #
            # Grad-CAM is only produced for successful skin predictions —
            # not for validation failures, healthy skin, OOD, or low-
            # confidence results (those branches in
            # ``get_recommendation`` already set ``gradcam_image: null``).
            pred_idx = result.get("pred_idx")
            wants_gradcam = (
                result.get("success", False) is True
                and pred_idx is not None
                and result.get("gradcam_image") is None
            )
            if wants_gradcam:
                background_tasks.add_task(
                    _run_gradcam_background,
                    prediction_id,
                    str(temp_path),
                    int(pred_idx),
                    float(confidence),
                )
            else:
                # No Grad-CAM to generate; release the temp file now.
                try:
                    temp_path.unlink(missing_ok=True)
                except Exception:
                    pass
                get_job_store().mark_no_gradcam(prediction_id)

            # Record prediction in history if user is authenticated
            if current_user:
                background_tasks.add_task(
                    _save_history_background,
                    current_user.id,
                    result,
                    getattr(current_user, "email", None),
                    getattr(current_user, "full_name", None),
                )

            return result

        # ==================================================
        # Medicinal Image in Disease Module — Reject
        # ==================================================

        elif image_type == "Medicinal":
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass
            get_job_store().mark_no_gradcam(prediction_id)
            return {
                "success": False,
                "prediction_id": prediction_id,
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
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass
            get_job_store().mark_no_gradcam(prediction_id)
            return {
                "success": False,
                "prediction_id": prediction_id,
                "image_type": "Other",
                "classifier_confidence": confidence,
                "classifier_ood_scores": image_result.get("ood_scores", {}),
                "classifier_is_ood": image_result.get("is_ood", False),
                "message": (
                    "Unsupported image detected.\n"
                    "Please upload a skin disease image in the disease identification module."
                ),
            }

    except AttributeError as e:
        # The universal classifier's constructor swallows checkpoint-load
        # failures and sets self.inference.model = None. The AttributeError
        # surfaces later when predict() tries to call .eval() on it. Catch
        # that here and return a structured 503 instead of a confusing 500.
        msg = str(e)
        if "NoneType" in msg and "eval" in msg:
            logger = get_logger(__name__)
            logger.error(
                "Universal classifier model not loaded",
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
            get_job_store().mark_failed(prediction_id, "MODEL_LOAD_ERROR")
            raise ModelLoadError(
                model_path="ai/image_classifier/checkpoints/best_model.pth",
                reason="Universal classifier checkpoint not loaded",
            )
        # Re-raise any other AttributeError unchanged.
        raise
    except FileNotFoundError as e:
        # Most commonly: a model checkpoint file is missing from disk.
        # Surface this as a structured 503 so the browser can show a
        # real message instead of a generic "Network Error" (which is
        # what axios reports when a 500 response lacks CORS headers).
        logger = get_logger(__name__)
        logger.error(
            "Model checkpoint missing",
            err=str(e),
            missing_path=str(e.filename) if getattr(e, "filename", None) else None,
        )
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass
        get_job_store().mark_failed(prediction_id, "MODEL_LOAD_ERROR")
        raise ModelLoadError(
            model_path=str(getattr(e, "filename", "ai/checkpoints/")),
            reason=str(e),
        )
    except Exception as e:
        logger = get_logger(__name__)
        logger.error("Prediction failed", err=str(e), error_type=type(e).__name__)
        # Best-effort cleanup and job-store notification.
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass
        get_job_store().mark_failed(prediction_id, type(e).__name__)
        raise ModelError(
            message=f"Prediction failed: {e}",
            details={"image_type": image_type},
        )

    # Note: when the skin branch schedules a background task, control
    # returns from this function BEFORE the temp file is unlinked. The
    # background task itself owns the cleanup (see
    # ``_run_gradcam_background``). All other branches unlink eagerly
    # before returning.
