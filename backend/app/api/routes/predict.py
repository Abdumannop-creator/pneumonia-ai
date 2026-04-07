"""
Prediction API endpoint.
Accepts an X-ray image, runs AI inference, generates heatmap,
evaluates severity, and returns full diagnostic results.
"""

import json
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.config import UPLOAD_DIR
from app.db.crud import create_prediction
from app.schemas.prediction import PredictionOut
from app.services.inference import get_predictor
from app.services.severity import grade_severity
from app.services.gradcam import generate_heatmap_overlay, generate_demo_heatmap

router = APIRouter(prefix="/api", tags=["predict"])

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".dcm"}
HEATMAP_DIR = UPLOAD_DIR / "heatmaps"


@router.post("/predict", response_model=PredictionOut)
async def predict(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Upload an X-ray image and get pneumonia prediction results.
    
    Returns prediction label, confidence, severity grading,
    diagnostic recommendations, and Grad-CAM heatmap.
    """
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="Fayl nomi bo'sh")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Ruxsat etilmagan fayl turi: {ext}. Ruxsat: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Save uploaded file
    unique_name = f"{uuid.uuid4().hex}{ext}"
    save_path = UPLOAD_DIR / unique_name
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    try:
        contents = await file.read()
        save_path.write_bytes(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Faylni saqlashda xato: {e}")

    # Run AI inference
    predictor = get_predictor()
    try:
        result = predictor.predict(str(save_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI tahlilida xato: {e}")

    # Generate heatmap
    heatmap_data = {"heatmap_path": None, "heatmap_filename": None, "affected_area_percent": 0.0}

    if predictor.demo_mode:
        heatmap_data = generate_demo_heatmap(
            str(save_path), str(HEATMAP_DIR), result["prob_pneumonia"]
        )
    else:
        try:
            heatmap_data = generate_heatmap_overlay(
                model=predictor.model,
                model_name=predictor.meta["model_name"],
                input_tensor=result.get("input_tensor"),
                original_image_path=str(save_path),
                save_dir=str(HEATMAP_DIR),
                class_idx=1,  # PNEUMONIA class
            )
        except Exception as e:
            print(f"Heatmap generation failed, using demo: {e}")
            heatmap_data = generate_demo_heatmap(
                str(save_path), str(HEATMAP_DIR), result["prob_pneumonia"]
            )

    # Severity grading
    severity_data = grade_severity(result["confidence"], result["prob_pneumonia"])

    # Override affected_area_percent from heatmap if available
    if heatmap_data.get("affected_area_percent", 0) > 0:
        severity_data["affected_area_percent"] = heatmap_data["affected_area_percent"]

    # Save to database
    recommendations_json = json.dumps(severity_data["recommendations"], ensure_ascii=False)

    heatmap_stored_path = None
    if heatmap_data.get("heatmap_path"):
        heatmap_stored_path = str(
            Path(heatmap_data["heatmap_path"]).relative_to(UPLOAD_DIR.parent)
        )

    db_record = create_prediction(
        db,
        original_filename=file.filename,
        stored_path=str(save_path.relative_to(UPLOAD_DIR.parent)),
        prediction_label=result["prediction_label"],
        confidence=result["confidence"],
        prob_normal=result["prob_normal"],
        prob_pneumonia=result["prob_pneumonia"],
        model_name=result["model_name"],
        model_version=result["model_version"],
        preprocess_version=result["preprocess_version"],
        severity=severity_data["severity"],
        severity_level=severity_data["severity_level"],
        severity_description=severity_data["severity_description"],
        affected_area_percent=severity_data["affected_area_percent"],
        heatmap_path=heatmap_stored_path,
        recommendations=recommendations_json,
    )

    # Build response
    image_url = "/" + str(save_path.relative_to(UPLOAD_DIR.parent)).replace("\\", "/")
    heatmap_url = None
    if heatmap_stored_path:
        heatmap_url = "/" + heatmap_stored_path.replace("\\", "/")

    return PredictionOut(
        id=db_record.id,
        created_at=db_record.created_at,
        prediction_label=result["prediction_label"],
        confidence=result["confidence"],
        prob_normal=result["prob_normal"],
        prob_pneumonia=result["prob_pneumonia"],
        model_name=result["model_name"],
        model_version=result["model_version"],
        preprocess_version=result["preprocess_version"],
        image_url=image_url,
        severity=severity_data["severity"],
        severity_level=severity_data["severity_level"],
        severity_description=severity_data["severity_description"],
        severity_color=severity_data["severity_color"],
        affected_area_percent=severity_data["affected_area_percent"],
        heatmap_url=heatmap_url,
        recommendations=severity_data["recommendations"],
    )
