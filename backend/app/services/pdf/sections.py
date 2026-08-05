from reportlab.platypus import (
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image
)
from reportlab.lib import colors
from reportlab.lib.units import inch

from .styles import (
    title_style,
    heading_style,
    body_style,
    small_style,
    HEADER_COLOR,
    ROW_COLOR,
    BORDER_COLOR,
)


# ==========================================================
# Header
# ==========================================================

def add_header(story, report_id, date, time, logo_path=None):
    if logo_path:
        try:
            logo = Image(logo_path, width=0.7 * inch, height=0.7 * inch)
            story.append(logo)
        except Exception:
            pass

    story.append(Paragraph("HERBAL-AI", title_style))
    story.append(
        Paragraph("AI Skin Disease Analysis Report", heading_style)
    )

    info = [
        ["Report ID", report_id],
        ["Date", date],
        ["Time", time],
    ]

    table = Table(info, colWidths=[1.5 * inch, 4 * inch])

    table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ("BACKGROUND", (0, 0), (0, -1), HEADER_COLOR),
            ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
            ("BACKGROUND", (1, 0), (1, -1), ROW_COLOR),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ])
    )

    story.append(table)
    story.append(Spacer(1, 20))


# ==========================================================
# Prediction
# ==========================================================
from reportlab.platypus import Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.units import inch


def add_prediction(story, prediction, confidence, level):

    story.append(Paragraph("AI Prediction", heading_style))

    data = [
        ["Disease", prediction],
        ["Confidence", f"{confidence:.2f}%"],
        ["Confidence Level", level],
    ]

    table = Table(
        data,
        colWidths=[2.2 * inch, 3.5 * inch]
    )

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F0FDF4")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#15803d")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),

            ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#166534")),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),

            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 10),

            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )

    story.append(table)

    story.append(Spacer(1, 20))


# ==========================================================
# Images
# ==========================================================

def add_images(story, original_image, gradcam_image):

    story.append(Paragraph("Analysis Images", heading_style))

    try:

        original = Image(
            original_image,
            width=2.6 * inch,
            height=2.6 * inch
        )

        gradcam = Image(
            gradcam_image,
            width=2.6 * inch,
            height=2.6 * inch
        )

        table = Table(
            [
                ["Original Image", "Grad-CAM"],
                [original, gradcam]
            ]
        )

        table.setStyle(
            TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
                ("GRID", (0, 1), (-1, 1), 0.5, colors.grey),
            ])
        )

        story.append(table)

    except Exception:

        story.append(
            Paragraph(
                "Images unavailable.",
                body_style
            )
        )

    story.append(Spacer(1, 20))

# ==========================================================
# Disease Overview
# ==========================================================

def add_disease_overview(story, description, symptoms, prevention):
    story.append(Paragraph("Disease Overview", heading_style))

    story.append(
        Paragraph(f"<b>Description:</b> {description}", body_style)
    )

    story.append(
        Paragraph(f"<b>Symptoms:</b> {symptoms}", body_style)
    )

    story.append(
        Paragraph(f"<b>Prevention:</b> {prevention}", body_style)
    )

    story.append(Spacer(1, 20))


# ==========================================================
# Herbal Recommendations
# ==========================================================

def add_herbs(story, herbs):
    story.append(
        Paragraph("Recommended Herbs", heading_style)
    )

    herb_text = ", ".join(herbs)

    story.append(
        Paragraph(herb_text, body_style)
    )

    story.append(Spacer(1, 20))


# ==========================================================
# AI Summary
# ==========================================================

def add_summary(story, summary):

    story.append(
        Paragraph(
            "AI Medical Summary",
            heading_style
        )
    )

    table = Table(
        [
            [
                Paragraph(summary, body_style)
            ]
        ],
        colWidths=[6.2 * inch]
    )

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F9FAFB")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 12),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ])
    )

    story.append(table)

    story.append(
        Spacer(1, 20)
    )

# ==========================================================
# Disclaimer
# ==========================================================

def add_disclaimer(story):
    story.append(
        Paragraph("Medical Disclaimer", heading_style)
    )

    disclaimer = (
        "This report is generated using Artificial Intelligence "
        "for educational and informational purposes only. "
        "It should not be considered a medical diagnosis. "
        "Please consult a qualified dermatologist for "
        "professional medical advice."
    )

    story.append(
        Paragraph(disclaimer, small_style)
    )