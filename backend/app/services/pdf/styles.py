from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics

# ==========================================================
# Optional Font Registration
# ==========================================================

try:
    pdfmetrics.registerFont(
        TTFont(
            "Helvetica",
            "Helvetica.ttf"
        )
    )
except Exception:
    pass

styles = getSampleStyleSheet()

# ==========================================================
# Title
# ==========================================================

title_style = styles["Heading1"]

title_style.alignment = TA_CENTER

title_style.fontSize = 24

title_style.spaceAfter = 20

title_style.textColor = colors.HexColor("#166534")

# ==========================================================
# Heading
# ==========================================================

heading_style = styles["Heading2"]

heading_style.alignment = TA_LEFT

heading_style.textColor = colors.HexColor("#15803d")

heading_style.spaceAfter = 10

# ==========================================================
# Body
# ==========================================================

body_style = styles["BodyText"]

body_style.fontSize = 11

body_style.leading = 18

body_style.spaceAfter = 8

# ==========================================================
# Small Text
# ==========================================================

small_style = styles["BodyText"]

small_style.fontSize = 9

small_style.leading = 12

small_style.textColor = colors.grey

# ==========================================================
# Table Colors
# ==========================================================

HEADER_COLOR = colors.HexColor("#15803d")

ROW_COLOR = colors.whitesmoke

BORDER_COLOR = colors.HexColor("#d1d5db")