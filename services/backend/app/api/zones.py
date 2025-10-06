from __future__ import annotations

import logging
from datetime import date

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, root_validator, validator
from shapely.geometry import shape

from app.services.zones_job import job_status, start_job
from app.services.zones_runner import _run_zones_production

router = APIRouter(prefix="/zones", tags=["zones"])

logger = logging.getLogger(__name__)


class ProductionZonesRequest(BaseModel):
    aoi_geojson: dict = Field(..., description="Polygon or multipolygon AOI GeoJSON")
    aoi_name: str = Field(..., description="AOI name used in export prefixes")
    start_date: date = Field(..., description="Start date in YYYY-MM-DD format (inclusive)")
    end_date: date = Field(..., description="End date in YYYY-MM-DD format (inclusive)")
    n_classes: int = Field(5, ge=3, le=7, description="Number of NDVI zones to classify")

    @validator("aoi_geojson")
    def _validate_geojson(cls, value: dict) -> dict:
        try:
            geom = shape(value)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid AOI geometry: {exc}") from exc
        if geom.is_empty:
            raise ValueError("AOI geometry cannot be empty")
        if geom.geom_type not in {"Polygon", "MultiPolygon"}:
            raise ValueError("AOI must be a Polygon or MultiPolygon")
        return value

    @validator("aoi_name")
    def _validate_name(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("aoi_name cannot be blank")
        return trimmed

    @root_validator
    def _validate_dates(cls, values: dict) -> dict:
        start = values.get("start_date")
        end = values.get("end_date")
        if start and end and end < start:
            raise ValueError("end_date must be on or after start_date")
        return values


@router.post("/production", status_code=202)
def create_production_zones(request: ProductionZonesRequest):
    try:
        job_id = start_job(_run_zones_production, request)
    except ValueError as exc:
        logger.warning("Zone production request validation failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Zone production failed to start: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"job_id": job_id}


@router.get("/production/status/{job_id}")
def production_status(job_id: str):
    status = job_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return status
