from __future__ import annotations

import logging
import os
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.products import router as products_router
from app.api.tiles import router as tiles_router
from app.services.earth_engine import ensure_ee

logger = logging.getLogger(__name__)

app = FastAPI(title="Vision Backend", version="1.0.0")


@app.on_event("startup")
def _startup() -> None:
    try:
        ensure_ee()
    except Exception as exc:  # pragma: no cover - EE errors vary at runtime
        logger.error("Earth Engine initialisation failed: %s", exc)
        raise


cors_origins = os.getenv("CORS_ORIGINS", "*")
allowed_origins: List[str]
if cors_origins == "*":
    allowed_origins = ["*"]
else:
    allowed_origins = [origin.strip() for origin in cors_origins.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(products_router)
app.include_router(tiles_router)


@app.get("/")
def root() -> dict[str, object]:
    return {
        "service": "vision-backend",
        "project": os.getenv("GCP_PROJECT"),
        "region": os.getenv("GCP_REGION"),
    }


@app.get("/healthz")
def healthz() -> dict[str, object]:
    return {"ok": True}
