from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List


class PredictionOut(BaseModel):
    id: int
    created_at: datetime
    prediction_label: str
    confidence: float
    prob_normal: float
    prob_pneumonia: float
    model_name: str
    model_version: str
    preprocess_version: str
    image_url: str

    # Severity grading
    severity: Optional[str] = None
    severity_level: Optional[int] = None
    severity_description: Optional[str] = None
    severity_color: Optional[str] = None
    affected_area_percent: Optional[float] = 0.0

    # Heatmap
    heatmap_url: Optional[str] = None

    # Recommendations
    recommendations: Optional[List[str]] = None

    class Config:
        from_attributes = True