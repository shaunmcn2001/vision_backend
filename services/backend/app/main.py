from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import Settings
from .api.routes import router as api_router

settings = Settings()

app = FastAPI(title="Agri NDVI Backend", version="0.1.0", docs_url="/docs", redoc_url="/redoc")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healths")
def healthz():
    return {"ok": True, "project": settings.GCP_PROJECT, "region": settings.GCP_REGION}

@app.get("/")
def root():
    return {"service": "agri-backend", "status": "running"}

app.include_router(api_router, prefix="/api")
