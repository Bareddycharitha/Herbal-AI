from pydantic import BaseModel
from typing import List, Optional


# ==========================================================
# Prediction
# ==========================================================

class Prediction(BaseModel):
    disease: str
    confidence: float


# ==========================================================
# Herb Information
# ==========================================================

class HerbRecommendation(BaseModel):
    name: str
    botanical_name: str
    family: str

    efficacy: int
    weight: float

    benefits: List[str]
    active_compounds: List[str]
    phytochemicals: List[str]

    preparation_method: str

    side_effects: List[str]
    contraindications: List[str]

    evidence_level: str
    research_papers: List[str]

    skin_types: List[str]


# ==========================================================
# Top Prediction
# ==========================================================

class TopPrediction(BaseModel):
    disease: str
    confidence: float


# ==========================================================
# Complete API Response
# ==========================================================

class PredictionResponse(BaseModel):

    success: bool

    prediction: Prediction

    recommendation_level: str

    warning: Optional[str] = None

    top_predictions: List[TopPrediction]

    recommended_herbs: List[HerbRecommendation]