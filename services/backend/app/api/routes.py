from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import ee, os
from ee import ServiceAccountCredentials
from app.services.gcs import download_json, exists
from app.services.ndvi import get_or_compute_and_cache_ndvi, gcs_ndvi_path

router = APIRouter()

# ---- EE init via Service Account (no interactive auth) ----
SA_EMAIL = "ee-agri-worker@baradine-farm.iam.gserviceaccount.com"
KEY_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")  # e.g. /opt/render/project/src/ee-key.json

def init_ee():
    if not KEY_PATH:
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS env var is not set")
    creds = ServiceAccountCredentials(SA_EMAIL, KEY_PATH)
    ee.Initialize(credentials=creds, opt_url="https://earthengine.googleapis.com")

# Simple ping endpoint
@router.get("/ping")
def ping():
    return {"message": "pong"}

# Earth Engine health check
@router.get("/ee/health")
def ee_health():
    try:
        init_ee()
        # trivial call to confirm connectivity
        _ = ee.Date(0).format().getInfo()
        return {"ok": True, "project": os.getenv("GCP_PROJECT")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"EE init failed: {str(e)}")

# Schema for NDVI request
class NDVIRequest(BaseModel):
    geometry: dict
    start: str
    end: str
    collection: str = "SENTINEL/2_SR_HARMONIZED"
    scale: int = 10

# NDVI monthly endpoint (simplified for testing)
@router.post("/ndvi/monthly")
def ndvi_monthly(req: NDVIRequest):
    try:
        init_ee()

        geom = ee.Geometry(req.geometry)
        collection = (
            ee.ImageCollection(req.collection)
            .filterBounds(geom)
            .filterDate(req.start, req.end)
        )

        def add_ndvi(img):
            ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
            return img.addBands(ndvi)

        with_ndvi = collection.map(add_ndvi)

        # compute per-calendar-month means across the requested period
        months = ee.List.sequence(1, 12)
        results = []
        for m in months.getInfo():
            monthly = with_ndvi.filter(ee.Filter.calendarRange(m, m, "month")).mean().select("NDVI")
            value = monthly.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geom,
                scale=req.scale,
                bestEffort=True,
            ).get("NDVI").getInfo()
            results.append({"month": int(m), "ndvi": value})

        return {"ok": True, "data": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NDVI failed: {str(e)}")
