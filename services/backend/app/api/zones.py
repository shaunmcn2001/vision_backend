from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
from shapely.geometry import shape

from app.services import zones as zone_service


router = APIRouter(prefix="/zones", tags=["zones"])


class ProductionZonesRequest(BaseModel):
    aoi_geojson: dict = Field(..., description="Polygon or multipolygon AOI GeoJSON")
    aoi_name: str = Field(..., description="AOI name used in export prefixes")
    months: List[str] = Field(..., description="Months in YYYY-MM format")
    indices_for_zoning: List[str] = Field(
        default_factory=lambda: list(zone_service.DEFAULT_ZONE_INDICES),
        description="Indices to include when clustering zones",
    )
    cloud_prob_max: int = Field(zone_service.DEFAULT_CLOUD_PROB_MAX, ge=0, le=100)
    k_zones: int = Field(zone_service.DEFAULT_K_ZONES, ge=2)
    cv_mask_threshold: float = Field(zone_service.DEFAULT_CV_THRESHOLD, ge=0)
    min_mapping_unit_ha: float = Field(zone_service.DEFAULT_MIN_MAPPING_UNIT_HA, gt=0)
    include_zonal_stats: bool = Field(True, description="Export per-zone statistics CSV")
    sample_size: int = Field(zone_service.DEFAULT_SAMPLE_SIZE, gt=0)

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

    @validator("months")
    def _validate_months(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("At least one month must be provided")
        seen = []
        for month in value:
            month_str = str(month).strip()
            if len(month_str) != 7 or month_str[4] != "-":
                raise ValueError(f"Invalid month format: {month}")
            year_part, month_part = month_str.split("-")
            if not (year_part.isdigit() and month_part.isdigit()):
                raise ValueError(f"Invalid month value: {month}")
            if int(month_part) < 1 or int(month_part) > 12:
                raise ValueError(f"Month must be between 01 and 12: {month}")
            if month_str not in seen:
                seen.append(month_str)
        return seen


@router.post("/production")
def create_production_zones(request: ProductionZonesRequest):
    try:
        artifacts = zone_service.build_zone_artifacts(
            request.aoi_geojson,
            months=request.months,
            indices_for_zoning=request.indices_for_zoning,
            cloud_prob_max=request.cloud_prob_max,
            k_zones=request.k_zones,
            cv_mask_threshold=request.cv_mask_threshold,
            min_mapping_unit_ha=request.min_mapping_unit_ha,
            sample_size=request.sample_size,
            include_stats=request.include_zonal_stats,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        bucket = zone_service.resolve_export_bucket()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    tasks = zone_service.start_zone_exports(
        artifacts,
        aoi_name=request.aoi_name,
        months=request.months,
        bucket=bucket,
        include_stats=request.include_zonal_stats,
    )

    prefix = zone_service.export_prefix(request.aoi_name, request.months)
    stats_prefix = prefix + "_zonal_stats"
    start_month, end_month = zone_service.month_bounds(request.months)

    def _task_payload(task):
        if task is None:
            return None
        return {"id": getattr(task, "id", None)}

    return {
        "bucket": bucket,
        "paths": {
            "raster": f"gs://{bucket}/{prefix}.tif",
            "vectors": f"gs://{bucket}/{prefix}",
            "zonal_stats": f"gs://{bucket}/{stats_prefix}.csv" if request.include_zonal_stats else None,
        },
        "tasks": {
            "raster": _task_payload(tasks["raster"]),
            "vectors": _task_payload(tasks["vectors"]),
            "zonal_stats": _task_payload(tasks["stats"]),
        },
        "metadata": {
            "aoi_name": request.aoi_name,
            "months": request.months,
            "month_start": start_month,
            "month_end": end_month,
            "k_zones": request.k_zones,
        },
    }

