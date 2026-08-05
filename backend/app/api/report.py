"""
PDF Report Generation API Endpoint

Generates professional medical reports with predictions, images, and AI summaries.
"""

import json
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.concurrency import run_in_threadpool

from backend.app.config import get_settings, Settings
from backend.app.dependencies import get_current_active_user
from backend.app.schemas.report import ReportData
from backend.app.services.pdf.report_generator import generate_report
from backend.app.utils.file_validator import file_validator, generate_secure_temp_path
from backend.app.exceptions import FileValidationError, ModelError

router = APIRouter(
    prefix="/report",
    tags=["PDF Report"]
)


@router.post("/")
async def create_report(
    image: UploadFile = File(...),
    report_data: str = Form(...),
    settings: Settings = Depends(get_settings),
    current_user = Depends(get_current_active_user),
):
    """
    Generate a PDF report for a skin disease analysis.

    Requires:
    - image: The original uploaded image
    - report_data: JSON string containing prediction, disease info, herbal recommendations, summary, and gradcam image
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
    # Parse Report Data
    # ======================================================

    try:
        data = ReportData.model_validate(json.loads(report_data))
    except Exception as e:
        from backend.app.exceptions import ValidationError
        raise ValidationError(
            message="Invalid report data format",
            details={"error": str(e)},
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

    try:
        # ==================================================
        # Generate Report
        # ==================================================

        pdf_buffer = await run_in_threadpool(
            generate_report,
            prediction_result=data.prediction,
            disease_info=data.disease_info,
            herbal_info=data.herbal_recommendations,
            ai_summary=data.summary,
            original_image=str(temp_path),
            gradcam_image=data.gradcam_image,
        )

        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition": 'attachment; filename="HerbalAI_Report.pdf"'
            },
        )

    except Exception as e:
        from backend.app.utils.logging import get_logger
        logger = get_logger(__name__)
        logger.error("Report generation failed", error=str(e), error_type=type(e).__name__)
        raise ModelError(
            message=f"Report generation failed: {e}",
        )

    finally:
        # Cleanup temp file
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass