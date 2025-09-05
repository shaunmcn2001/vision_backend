from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router
from fastapi import Request, HTTPException
import os

app = FastAPI(
    title="Agri NDVI Backend",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Allow all origins (adjust later if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "service": "vision-backend",
        "project": os.getenv("GCP_PROJECT"),
        "region": os.getenv("GCP_REGION")
    }

@app.get("/")
def root():
    return {"service": "vision-backend", "status": "running"}
    
@app.middleware("http")
async def require_api_key(req: Request, call_next):
    if req.url.path.startswith(("/docs","/healthz")):  # allow health/docs
        return await call_next(req)
    if req.headers.get("x-api-key") != os.getenv("API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return await call_next(req)
    
# Mount API routes
app.include_router(api_router, prefix="/api")
