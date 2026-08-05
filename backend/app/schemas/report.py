from pydantic import BaseModel
from typing import List, Dict, Any


class ReportData(BaseModel):
    prediction: Dict[str, Any]
    disease_info: Dict[str, Any]
    herbal_recommendations: List[Dict[str, Any]]
    summary: str
    gradcam_image: str | None = None