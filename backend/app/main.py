import sys
from types import ModuleType

# Mock lzma before any other imports to fix ModuleNotFoundError: No module named '_lzma'
# This is required because torchvision and other ML libs import lzma internally.
if "lzma" not in sys.modules:
    lzma_mock = ModuleType("lzma")
    lzma_mock.LZMAError = Exception
    lzma_mock.CHECK_NONE = 0
    lzma_mock.CHECK_CRC32 = 1
    lzma_mock.CHECK_CRC64 = 4
    lzma_mock.CHECK_SHA256 = 10
    lzma_mock.open = lambda *args, **kwargs: None
    
    sys.modules["lzma"] = lzma_mock
    sys.modules["_lzma"] = lzma_mock

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from app.config import settings, UPLOAD_DIR
from app.api.routes.health import router as health_router
from app.api.routes.history import router as history_router
from app.api.routes.predict import router as predict_router
from app.db.init_db import init_db

app = FastAPI(title=settings.app_name)

# CORS middleware — allow frontend to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
(UPLOAD_DIR / "heatmaps").mkdir(parents=True, exist_ok=True)

# Mount static uploads
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# Mount frontend static files
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")

# Include routers
app.include_router(health_router)
app.include_router(predict_router)
app.include_router(history_router)


# Serve frontend index.html at root
@app.get("/", include_in_schema=False)
async def serve_frontend():
    from fastapi.responses import FileResponse
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Pneumonia AI API is running. Frontend not found."}


# Initialize database on startup
@app.on_event("startup")
def on_startup():
    init_db()
    print("=" * 60)
    print("  Pneumonia AI — Server ishga tushdi!")
    print("  Frontend:  http://localhost:8000")
    print("  API Docs:  http://localhost:8000/docs")
    print("=" * 60)