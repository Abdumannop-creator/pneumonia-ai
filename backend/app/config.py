from pydantic import BaseModel
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
DB_PATH = BASE_DIR / "app.db"

class Settings(BaseModel):
    app_name: str = "Pneumonia AI API"
    upload_dir: str = str(UPLOAD_DIR)
    db_url: str = f"sqlite:///{DB_PATH}"

settings = Settings()