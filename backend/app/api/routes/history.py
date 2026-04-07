import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.db.crud import list_predictions, get_prediction
from app.schemas.prediction import PredictionOut
from app.services.severity import grade_severity

router = APIRouter(prefix="/api", tags=["history"])


def _build_prediction_out(r) -> PredictionOut:
    """Convert a DB Prediction row to a PredictionOut schema."""
    image_url = "/" + r.stored_path.replace("\\", "/")

    heatmap_url = None
    if r.heatmap_path:
        heatmap_url = "/" + r.heatmap_path.replace("\\", "/")

    recommendations = None
    if r.recommendations:
        try:
            recommendations = json.loads(r.recommendations)
        except (json.JSONDecodeError, TypeError):
            recommendations = []

    # Get severity color from grading function
    severity_info = grade_severity(r.confidence, r.prob_pneumonia)
    severity_color = severity_info.get("severity_color", "#6b7280")

    return PredictionOut(
        id=r.id,
        created_at=r.created_at,
        prediction_label=r.prediction_label,
        confidence=r.confidence,
        prob_normal=r.prob_normal,
        prob_pneumonia=r.prob_pneumonia,
        model_name=r.model_name,
        model_version=r.model_version,
        preprocess_version=r.preprocess_version,
        image_url=image_url,
        severity=r.severity,
        severity_level=r.severity_level,
        severity_description=r.severity_description,
        severity_color=severity_color,
        affected_area_percent=r.affected_area_percent or 0.0,
        heatmap_url=heatmap_url,
        recommendations=recommendations,
    )


@router.get("/history", response_model=list[PredictionOut])
def history(limit: int = 50, offset: int = 0, db: Session = Depends(get_db)):
    rows = list_predictions(db, limit=limit, offset=offset)
    return [_build_prediction_out(r) for r in rows]


@router.get("/history/{prediction_id}", response_model=PredictionOut)
def history_one(prediction_id: int, db: Session = Depends(get_db)):
    r = get_prediction(db, prediction_id)
    if not r:
        raise HTTPException(status_code=404, detail="Topilmadi")
    return _build_prediction_out(r)