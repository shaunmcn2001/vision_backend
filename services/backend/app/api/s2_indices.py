"""API routes for Sentinel-2 index exports."""
from __future__ import annotations

from datetime import datetime
from typing import List, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
from shapely.geometry import shape

from app import exports, indices

router = APIRouter(prefix="/export/s2", tags=["sentinel-2"])


class Sentinel2ExportRequest(BaseModel):
    aoi_geojson: dict = Field(..., description="GeoJSON polygon or multipolygon AOI")
    months: List[str] = Field(..., description="Months to export in YYYY-MM format")
    indices: List[str] = Field(..., description="Sentinel-2 index names")
    export_target: Literal["drive", "gcs", "zip"] = Field(
        "zip", description="Destination for exports"
    )
    aoi_name: str = Field(..., description="Name of the AOI used in filenames")
    scale_m: int = Field(10, description="Target output resolution in metres")
    cloud_prob_max: int = Field(40, description="Maximum S2 cloud probability to retain")

    @validator("aoi_geojson")
    def _validate_geojson(cls, value: dict) -> dict:
        try:
            geom = shape(value)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid GeoJSON: {exc}") from exc
        if geom.is_empty:
            raise ValueError("AOI geometry is empty")
        if geom.geom_type not in {"Polygon", "MultiPolygon"}:
            raise ValueError("AOI must be a Polygon or MultiPolygon")
        return value

    @validator("months")
    def _validate_months(cls, values: List[str]) -> List[str]:
        if not values:
            raise ValueError("At least one month must be provided")
        seen = []
        for value in values:
            try:
                datetime.strptime(value, "%Y-%m")
            except ValueError as exc:
                raise ValueError(f"Invalid month format: {value}") from exc
            if value not in seen:
                seen.append(value)
        return seen

    @validator("indices")
    def _validate_indices(cls, values: List[str]) -> List[str]:
        if not values:
            raise ValueError("At least one index must be specified")
        normalised = []
        for value in values:
            upper = value.upper()
            if upper not in indices.SUPPORTED_INDICES:
                raise ValueError(f"Unsupported index: {value}")
            if upper not in normalised:
                normalised.append(upper)
        return normalised

    @validator("scale_m")
    def _validate_scale(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("scale_m must be positive")
        return value

    @validator("cloud_prob_max")
    def _validate_cloud_prob(cls, value: int) -> int:
        if value < 0 or value > 100:
            raise ValueError("cloud_prob_max must be between 0 and 100")
        return value

    @validator("aoi_name")
    def _validate_name(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("aoi_name cannot be empty")
        return trimmed


@router.post("/indices")
def start_export(request: Sentinel2ExportRequest):
    try:
        job = exports.create_job(
            aoi_geojson=request.aoi_geojson,
            months=request.months,
            index_names=request.indices,
            export_target=request.export_target,
            aoi_name=request.aoi_name,
            scale_m=request.scale_m,
            cloud_prob_max=request.cloud_prob_max,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to queue export: {exc}") from exc

    return {"job_id": job.job_id, "state": job.state}


@router.get("/indices/{job_id}/status")
def get_status(job_id: str):
    status = exports.job_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return status


@router.get("/indices/{job_id}/download")
def download(job_id: str):
    job = exports.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.export_target == "zip":
        try:
            zip_path = exports.get_zip_path(job_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        def iterator():
            with zip_path.open("rb") as fh:
                for chunk in iter(lambda: fh.read(8192), b""):
                    yield chunk

        headers = {
            "Content-Disposition": f'attachment; filename="{zip_path.name}"'
        }
        return StreamingResponse(iterator(), media_type="application/zip", headers=headers)

    # For Drive or GCS return the status payload (contains destination URIs / signed URLs)
    status = exports.job_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(status)

