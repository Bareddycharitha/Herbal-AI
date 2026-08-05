import os
from datetime import datetime

from io import BytesIO
from reportlab.platypus import SimpleDocTemplate
from reportlab.lib.pagesizes import A4

from .sections import (
    add_header,
    add_prediction,
    add_images,
    add_disease_overview,
    add_herbs,
    add_summary,
    add_disclaimer,
)


REPORTS_DIR = "backend/app/services/pdf/reports"
LOGO_PATH = "backend/app/services/pdf/assets/logo.png"


def generate_report(
    prediction_result: dict,
    disease_info: dict,
    herbal_info: list,
    ai_summary: str,
    original_image: str,
    gradcam_image: str,
):
    """
    Generates a PDF report and returns it as a BytesIO object.
    """

    os.makedirs(REPORTS_DIR, exist_ok=True)

    timestamp = datetime.now()

    report_id = timestamp.strftime("HAI-%Y%m%d-%H%M%S")

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=30,
        leftMargin=30,
        topMargin=30,
        bottomMargin=30,
    )

    story = []

    # --------------------------------------------------
    # Header
    # --------------------------------------------------

    add_header(
        story=story,
        report_id=report_id,
        date=timestamp.strftime("%d %B %Y"),
        time=timestamp.strftime("%I:%M %p"),
        logo_path=LOGO_PATH if os.path.exists(LOGO_PATH) else None,
    )

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    disease = prediction_result["disease"]

    add_prediction(
        story=story,
        prediction=disease,
        confidence=prediction_result["confidence"],
        level=prediction_result["confidence_level"],
    )

    # --------------------------------------------------
    # Images
    # --------------------------------------------------

    add_images(
        story,
        original_image,
        gradcam_image,
    )

    # ==================================================
    # HEALTHY SKIN REPORT
    # ==================================================

    if disease == "Healthy Skin":

        add_disease_overview(
            story,
            description="No visible skin disease was detected.",
            symptoms=[],
            prevention=[
                "Cleanse your skin regularly.",
                "Use sunscreen daily.",
                "Moisturize when required.",
                "Drink enough water.",
                "Consult a dermatologist if unusual symptoms develop.",
            ],
        )

    # ==================================================
    # DISEASE REPORT
    # ==================================================

    else:

        add_disease_overview(
            story,
            description=disease_info.get("description", "N/A"),
            symptoms=disease_info.get("symptoms", []),
            prevention=disease_info.get("prevention", []),
        )

        herb_names = [
            herb.get("name", "")
            for herb in herbal_info
        ]

        add_herbs(
            story,
            herb_names,
        )

    # --------------------------------------------------
    # AI Summary
    # --------------------------------------------------

    add_summary(
        story,
        ai_summary,
    )

    # --------------------------------------------------
    # Disclaimer
    # --------------------------------------------------

    add_disclaimer(
        story,
    )

    # --------------------------------------------------
    # Build PDF
    # --------------------------------------------------

    document.build(story)

    buffer.seek(0)

    return buffer

    