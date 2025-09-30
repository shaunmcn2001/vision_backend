from datetime import date
from typing import Any, Literal, get_args

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import ee, os
from app.services.gcs import bucket_name, download_json, exists, sign_url
from app.services.indices import SUPPORTED_INDICES
from app.services.ndvi import (
    DEFAULT_COLLECTION,
    DEFAULT_SCALE,
    compute_monthly_index,
    get_or_compute_and_cache_index,
    gcs_index_csv_path,
    gcs_index_path,
    list_cached_years,
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

SupportedIndex = Literal["ndvi", "evi", "gndvi", "ndre"]

if set(SUPPORTED_INDICES) != set(get_args(SupportedIndex)):
    raise RuntimeError(
        "Supported index literal must match definitions in app.services.indices."
    )


class IndexSelection(BaseModel):
    code: SupportedIndex
    parameters: dict[str, Any] = Field(default_factory=dict)


class MonthlyIndexRequest(BaseModel):
    geometry: dict
    start: str
    end: str
    collection: str = DEFAULT_COLLECTION
    scale: int = DEFAULT_SCALE
    index: IndexSelection = Field(default_factory=lambda: IndexSelection(code="ndvi"))


@router.post("/ndvi/monthly")
def ndvi_monthly(req: MonthlyIndexRequest):
    try:
        start_date = date.fromisoformat(req.start[:10])
        end_date = date.fromisoformat(req.end[:10])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid date: {exc}") from exc

    try:
        result = compute_monthly_index(
            req.geometry,
            start=start_date,
            end=end_date,
            index_code=req.index.code,
            collection=req.collection,
            scale=req.scale,
            parameters=req.index.parameters,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=500, detail=f"Index computation failed: {str(exc)}"
        ) from exc

    return {"ok": True, **result}

@router.get("/ndvi/cache/{field_id}/{year}")
def ndvi_cache_get(field_id: str, year: int, index: SupportedIndex = Query("ndvi")):
    path = gcs_index_path(index, field_id, year)
    if not exists(path):
        raise HTTPException(status_code=404, detail="No cached index for this field/year")
    return download_json(path)

@router.post("/ndvi/monthly/by-field/{field_id}")
def ndvi_monthly_by_field(
    field_id: str,
    year: int,
    index: SupportedIndex = Query("ndvi"),
    force: bool = False,
):
    """
    Compute or return cached monthly vegetation index for a field and year.
    Caches to GCS at index-results/{index}/{field_id}/{year}.json
    """
    geom_path = f"fields/{field_id}/field.geojson"
    if not exists(geom_path):
        raise HTTPException(status_code=404, detail="Field not found")
    geometry = download_json(geom_path)
    return get_or_compute_and_cache_index(
        field_id,
        geometry,
        year,
        index_code=index,
        force=force,
    )

@router.get("/ndvi/years/{field_id}")
def ndvi_years(field_id: str, index: SupportedIndex = Query("ndvi")):
    return {
        "field_id": field_id,
        "index": index,
        "years": list_cached_years(field_id, index),
    }

@router.get("/ndvi/links/{field_id}/{year}")
def ndvi_links(field_id: str, year: int, index: SupportedIndex = Query("ndvi")):
    json_path = gcs_index_path(index, field_id, year)
    csv_path = gcs_index_csv_path(index, field_id, year)
    bucket = bucket_name()
    return {
        "index": index,
        "json": {
            "gs": f"gs://{bucket}/{json_path}",
            "signed": sign_url(json_path),
        },
        "csv": {
            "gs": f"gs://{bucket}/{csv_path}",
            "signed": sign_url(csv_path),
        },
    }

router.include_router(export_router, prefix="/export", tags=["export"])
