from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from datetime import datetime

from app.db.session import Base

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    original_filename = Column(String(255), nullable=False)
    stored_path = Column(String(500), nullable=False)

    prediction_label = Column(String(20), nullable=False)
    confidence = Column(Float, nullable=False)

    prob_normal = Column(Float, nullable=False)
    prob_pneumonia = Column(Float, nullable=False)

    model_name = Column(String(100), nullable=False)
    model_version = Column(String(100), nullable=False)
    preprocess_version = Column(String(100), nullable=False)

    # Severity grading
    severity = Column(String(20), nullable=True)
    severity_level = Column(Integer, nullable=True)
    severity_description = Column(String(500), nullable=True)
    affected_area_percent = Column(Float, nullable=True, default=0.0)

    # Heatmap
    heatmap_path = Column(String(500), nullable=True)

    # Recommendations (stored as JSON string)
    recommendations = Column(Text, nullable=True)

    request_meta = Column(Text, nullable=True)  # optional JSON string