from sqlalchemy.orm import Session
from app.db.models import Prediction

def create_prediction(db: Session, **kwargs) -> Prediction:
    obj = Prediction(**kwargs)
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj

def list_predictions(db: Session, limit: int = 50, offset: int = 0):
    return (
        db.query(Prediction)
        .order_by(Prediction.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

def get_prediction(db: Session, prediction_id: int):
    return db.query(Prediction).filter(Prediction.id == prediction_id).first()