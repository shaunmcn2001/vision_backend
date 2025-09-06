from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router
from app.api.fields import router as fields_router
from app.api.fields_upload import router as fields_upload_router
from app.api.tiles import router as tiles_router
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
        "region": os.getenv("GCP_REGION"),
    }

@app.get("/")
def root():
    return {"service": "vision-backend", "status": "running"}

# Simple API key middleware (protects everything except health/docs)
@app.middleware("http")
async def require_api_key(req: Request, call_next):
    public_paths = (
        "/healthz",
        "/docs",
        "/redoc",
        "/openapi.json",           # needed for Swagger UI to load
        "/docs/oauth2-redirect",   # docs assets
        "/docs/swagger-ui",
        "/docs/swagger-ui-init.js",
        "/docs/swagger-ui-bundle.js",
        "/docs/swagger-ui.css",
        "/",
    )
    if any(req.url.path.startswith(p) for p in public_paths):
        return await call_next(req)
    if req.headers.get("x-api-key") != os.getenv("API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return await call_next(req)

# Mount API routes
app.include_router(api_router, prefix="/api")          # existing EE/NDVI routes
app.include_router(fields_router, prefix="/api/fields")  # new Fields CRUD (GCS-backed)
app.include_router(api_router, prefix="/api")                 # EE/NDVI routes
app.include_router(fields_router, prefix="/api/fields")       # existing JSON-based fields API
app.include_router(fields_upload_router, prefix="/api/fields")# new file upload endpoint
app.include_router(tiles_router, prefix="/api")
