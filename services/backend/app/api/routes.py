from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import ee, os

router = APIRouter()

# Simple ping endpoint
@router.get("/ping")
def ping():
    return {"message": "pong"}

# Earth Engine health check
@router.get("/ee/health")
def ee_health():
    try:
        ee.Initialize(opt_url='https://earthengine.googleapis.com')
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
        ee.Initialize(opt_url='https://earthengine.googleapis.com')

        geom = ee.Geometry(req.geometry)
        collection = ee.ImageCollection(req.collection) \
            .filterBounds(geom) \
            .filterDate(req.start, req.end)

        # Example: NDVI calculation
        def add_ndvi(img):
            ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
            return img.addBands(ndvi)

        with_ndvi = collection.map(add_ndvi)

        # Monthly mean NDVI
        months = ee.List.sequence(1, 12)
        yearly = []

        for m in months.getInfo():
            monthly = with_ndvi.filter(ee.Filter.calendarRange(m, m, 'month')) \
                .mean().select('NDVI')
            value = monthly.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=geom, scale=req.scale
            ).get('NDVI').getInfo()
            yearly.append({"month": m, "ndvi": value})

        return {"ok": True, "data": yearly}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NDVI failed: {str(e)}")
