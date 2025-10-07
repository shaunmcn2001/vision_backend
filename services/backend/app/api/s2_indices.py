"""API routes for Sentinel-2 index exports."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import List, Literal

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
from shapely.geometry import shape
from starlette.background import BackgroundTask

from app import exports, indices
from app.services import zones as zone_service
from app.utils.geometry import area_ha
from app.utils.shapefile import shapefile_zip_to_geojson

router = APIRouter(prefix="/export/s2", tags=["sentinel-2"])

EVICTED_JOB_MESSAGE = "Export job is no longer available. Please request a new export."


def _min_field_hectares() -> float:
    raw_value = os.getenv("MIN_FIELD_HA", "1.0")
    try:
        return float(raw_value)
    except ValueError:  # pragma: no cover - defensive fallback
        return 1.0


@router.post("/indices/aoi")
async def prepare_aoi_geometry(
    file: UploadFile = File(
        ..., description="Zipped ESRI Shapefile containing polygon features"
    ),
    aoi_name: str | None = Form(
        None, description="Optional AOI name to sanitise for export filenames"
    ),
    enforce_area: bool = Query(
        True, description="Reject uploads smaller than MIN_FIELD_HA hectares"
    ),
):
    original_filename = file.filename or ""
    filename = original_filename.lower()
    if not filename.endswith(".zip"):
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Upload a zipped shapefile (.zip).",
        )

    content = await file.read()

    try:
        geometry = shapefile_zip_to_geojson(content)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=400, detail=f"Failed to parse shapefile: {exc}"
        ) from exc

    try:
        area = area_ha(geometry)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Area calculation failed: {exc}"
        ) from exc

    min_area = _min_field_hectares()
    if enforce_area and area < min_area:
        raise HTTPException(
            status_code=400,
            detail=f"AOI area {area:.2f} ha is smaller than minimum {min_area} ha",
        )

    provided_name = aoi_name if isinstance(aoi_name, str) else None
    safe_name = exports.sanitize_name(
        (provided_name or Path(original_filename).stem or "aoi")
    )

    response = {
        "geometry": geometry,
        "aoi_name": safe_name,
        "area_ha": round(area, 4),
    }
    return response


class ProductionZoneOptions(BaseModel):
    enabled: bool = Field(
        False,
        description="When true, compute production zones for the requested months",
    )
    n_classes: int = Field(zone_service.DEFAULT_N_CLASSES, ge=3, le=7)
    cv_mask_threshold: float = Field(zone_service.DEFAULT_CV_THRESHOLD, ge=0)
    apply_stability_mask: bool = Field(
        True,
        description=(
            "Disable to skip the coefficient-of-variation stability mask when building zones"
        ),
    )
    mmu_ha: float = Field(zone_service.DEFAULT_MIN_MAPPING_UNIT_HA, gt=0)
    smooth_radius_m: float = Field(zone_service.DEFAULT_SMOOTH_RADIUS_M, ge=0)
    open_radius_m: float = Field(zone_service.DEFAULT_OPEN_RADIUS_M, ge=0)
    close_radius_m: float = Field(zone_service.DEFAULT_CLOSE_RADIUS_M, ge=0)
    simplify_tol_m: float = Field(zone_service.DEFAULT_SIMPLIFY_TOL_M, ge=0)
    simplify_buffer_m: float = Field(zone_service.DEFAULT_SIMPLIFY_BUFFER_M)


class Sentinel2ExportRequest(BaseModel):
    aoi_geojson: dict = Field(..., description="GeoJSON polygon or multipolygon AOI")
    months: List[str] = Field(..., description="Months to export in YYYY-MM format")
    indices: List[str] = Field(..., description="Sentinel-2 index names")
    export_target: Literal["drive", "gcs", "zip"] = Field(
        "zip", description="Destination for exports"
    )
    aoi_name: str = Field(..., description="Name of the AOI used in filenames")
    scale_m: int = Field(10, description="Target output resolution in metres")
    cloud_prob_max: int = Field(
        40, description="Maximum S2 cloud probability to retain"
    )
    production_zones: ProductionZoneOptions | None = Field(
        None, description="Optional production zone generation parameters"
    )

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
    def _validate_months(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("At least one month must be provided")
        seen = []
        for month in value:
            try:
                datetime.strptime(month, "%Y-%m")
            except ValueError as exc:
                raise ValueError(f"Invalid month format: {month}") from exc
            if month not in seen:
                seen.append(month)
        return seen

    @validator("indices")
    def _validate_indices(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("At least one index must be specified")
        canonical_lookup = {name.lower(): name for name in indices.SUPPORTED_INDICES}
        normalised: List[str] = []
        for index in value:
            key = str(index).strip()
            canonical = canonical_lookup.get(key.lower())
            if canonical is None:
                raise ValueError(f"Unsupported index: {index}")
            if canonical not in normalised:
                normalised.append(canonical)
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

    @validator("production_zones", pre=True)
    def _normalise_zone_options(cls, value):
        if value in (None, False, ""):
            return None
        if value is True:
            return {"enabled": True}
        if isinstance(value, ProductionZoneOptions):
            return value
        if isinstance(value, dict):
            if "enabled" not in value:
                return {**value, "enabled": True}
            return value
        raise ValueError("production_zones must be a boolean or object")


@router.post("/indices")
def start_export(request: Sentinel2ExportRequest):
    try:
        zone_config = None
        if request.production_zones and request.production_zones.enabled:
            zone_config = exports.ZoneExportConfig(
                n_classes=request.production_zones.n_classes,
                cv_mask_threshold=request.production_zones.cv_mask_threshold,
                apply_stability_mask=request.production_zones.apply_stability_mask,
                min_mapping_unit_ha=request.production_zones.mmu_ha,
                smooth_radius_m=request.production_zones.smooth_radius_m,
                open_radius_m=request.production_zones.open_radius_m,
                close_radius_m=request.production_zones.close_radius_m,
                simplify_tolerance_m=request.production_zones.simplify_tol_m,
                simplify_buffer_m=request.production_zones.simplify_buffer_m,
                include_stats=True,
            )

        job = exports.create_job(
            aoi_geojson=request.aoi_geojson,
            months=request.months,
            index_names=request.indices,
            export_target=request.export_target,
            aoi_name=request.aoi_name,
            scale_m=request.scale_m,
            cloud_prob_max=request.cloud_prob_max,
            zone_config=zone_config,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Earth Engine initialisation failed. "
                "Ensure GEE_SERVICE_ACCOUNT_JSON is configured: "
                f"{exc}"
            ),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Failed to queue export: {exc}"
        ) from exc

    return {"job_id": job.job_id, "state": job.state}


@router.get("/indices/{job_id}/status")
def get_status(job_id: str):
    status = exports.job_status(job_id)
    if status is None:
        if exports.was_job_evicted(job_id):
            raise HTTPException(status_code=410, detail=EVICTED_JOB_MESSAGE)
        raise HTTPException(status_code=404, detail="Job not found")
    return status


@router.get("/indices/{job_id}/download")
def download(job_id: str):
    job = exports.get_job(job_id)
    if job is None:
        if exports.was_job_evicted(job_id):
            raise HTTPException(status_code=410, detail=EVICTED_JOB_MESSAGE)
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

        headers = {"Content-Disposition": f'attachment; filename="{zip_path.name}"'}
        background = BackgroundTask(exports.cleanup_job_files, job)
        return StreamingResponse(
            iterator(),
            media_type="application/zip",
            headers=headers,
            background=background,
        )

    # For Drive or GCS return the status payload (contains destination URIs / signed URLs)
    status = exports.job_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(status)
