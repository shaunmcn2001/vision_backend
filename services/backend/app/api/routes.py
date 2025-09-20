from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import ee, os
from app.services.gcs import download_json, exists, sign_url
from app.services.ndvi import (
    DEFAULT_REDUCE_REGION_CRS,
    DEFAULT_REDUCE_REGION_SCALE,
    get_or_compute_and_cache_ndvi,
    gcs_ndvi_path,
    list_cached_years,
    reduce_region_sampling,
)
from app.services.tiles import init_ee
from .export import router as export_router

router = APIRouter()

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
    collection: str = "COPERNICUS/S2_SR_HARMONIZED"
    scale: float = DEFAULT_REDUCE_REGION_SCALE
    crs: str | None = DEFAULT_REDUCE_REGION_CRS

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
            monthly_coll = with_ndvi.filter(ee.Filter.calendarRange(m, m, "month"))
            if monthly_coll.size().eq(0).getInfo():
                results.append({"month": int(m), "ndvi": None})
                continue

            monthly = monthly_coll.mean().select("NDVI")
            value = monthly.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geom,
                **reduce_region_sampling(scale=req.scale, crs=req.crs),
            ).get("NDVI").getInfo()
            results.append({"month": int(m), "ndvi": value})

        return {"ok": True, "data": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NDVI failed: {str(e)}")

@router.get("/ndvi/cache/{field_id}/{year}")
def ndvi_cache_get(field_id: str, year: int):
    path = gcs_ndvi_path(field_id, year)
    if not exists(path):
        raise HTTPException(status_code=404, detail="No cached NDVI for this field/year")
    return download_json(path)

@router.post("/ndvi/monthly/by-field/{field_id}")
def ndvi_monthly_by_field(field_id: str, year: int, force: bool = False):
    """
    Compute or return cached monthly NDVI for a field and year.
    Caches to GCS at ndvi-results/{field_id}/{year}.json
    """
    geom_path = f"fields/{field_id}/field.geojson"
    if not exists(geom_path):
        raise HTTPException(status_code=404, detail="Field not found")
    geometry = download_json(geom_path)
    return get_or_compute_and_cache_ndvi(field_id, geometry, year, force=force)

@router.get("/ndvi/years/{field_id}")
def ndvi_years(field_id: str):
    return {"field_id": field_id, "years": list_cached_years(field_id)}

@router.get("/ndvi/links/{field_id}/{year}")
def ndvi_links(field_id: str, year: int):
    json_path = gcs_ndvi_path(field_id, year)
    csv_path = f"ndvi-results/{field_id}/{year}.csv"
    return {
        "json": {"gs": f"gs://{os.getenv('GCS_BUCKET')}/{json_path}", "signed": sign_url(json_path)},
        "csv":  {"gs": f"gs://{os.getenv('GCS_BUCKET')}/{csv_path}",  "signed": sign_url(csv_path)}
    }

router.include_router(export_router, prefix="/export", tags=["export"])
