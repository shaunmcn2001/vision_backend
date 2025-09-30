from __future__ import annotations

import os
from datetime import datetime
from typing import List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
from shapely.geometry import shape

from app.services import zones as zone_service


router = APIRouter(prefix="/zones", tags=["zones"])


class _BaseAOIRequest(BaseModel):
    aoi_geojson: dict = Field(..., description="Polygon or multipolygon AOI GeoJSON")
    aoi_name: str = Field(..., description="AOI name used in export prefixes")

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


class ProductionZonesRequest(_BaseAOIRequest):
    months: List[str] = Field(..., description="Months in YYYY-MM format")
    cloud_prob_max: int = Field(zone_service.DEFAULT_CLOUD_PROB_MAX, ge=0, le=100)
    n_classes: int = Field(zone_service.DEFAULT_N_CLASSES, ge=3, le=7)
    cv_mask_threshold: float = Field(zone_service.DEFAULT_CV_THRESHOLD, ge=0)
    mmu_ha: float = Field(zone_service.DEFAULT_MIN_MAPPING_UNIT_HA, gt=0)
    smooth_kernel_px: int = Field(zone_service.DEFAULT_SMOOTH_KERNEL_PX, ge=0)
    simplify_tol_m: float = Field(zone_service.DEFAULT_SIMPLIFY_TOL_M, ge=0)
    export_target: Literal["zip", "gcs", "drive"] = Field(
        "zip", description="Destination for exports"
    )
    gcs_bucket: Optional[str] = Field(None, description="Override GCS bucket for exports")
    gcs_prefix: Optional[str] = Field(
        None, description="Optional prefix before zones/ when exporting to GCS"
    )
    include_zonal_stats: bool = Field(True, description="Export per-zone statistics CSV")

    @validator("months")
    def _validate_months(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("At least one month must be provided")
        parsed: dict[str, datetime] = {}
        for month in value:
            month_str = str(month).strip()
            try:
                parsed.setdefault(month_str, datetime.strptime(month_str, "%Y-%m"))
            except ValueError as exc:
                raise ValueError(f"Invalid month format: {month}") from exc
        ordered = sorted(parsed.items(), key=lambda item: item[1])
        return [month for month, _ in ordered]


@router.post("/production")
def create_production_zones(request: ProductionZonesRequest):
    resolved_bucket: Optional[str] = request.gcs_bucket
    if request.export_target == "gcs":
        resolved_bucket = (
            (request.gcs_bucket or "").strip()
            or os.getenv("GEE_GCS_BUCKET")
            or os.getenv("GCS_BUCKET")
            or ""
        ).strip()
        if not resolved_bucket:
            raise HTTPException(
                status_code=400,
                detail="A GCS bucket must be provided when export_target is 'gcs'.",
            )

    try:
        result = zone_service.export_selected_period_zones(
            request.aoi_geojson,
            request.aoi_name,
            request.months,
            cloud_prob_max=request.cloud_prob_max,
            n_classes=request.n_classes,
            cv_mask_threshold=request.cv_mask_threshold,
            mmu_ha=request.mmu_ha,
            smooth_kernel_px=request.smooth_kernel_px,
            simplify_tol_m=request.simplify_tol_m,
            export_target=request.export_target,
            gcs_bucket=resolved_bucket,
            gcs_prefix=request.gcs_prefix,
            include_zonal_stats=request.include_zonal_stats,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    result.pop("artifacts", None)
    metadata = result.get("metadata", {}) or {}
    used_months: List[str] = metadata.get("used_months") or request.months
    ym_start = used_months[0]
    ym_end = used_months[-1]

    response = {
        "ok": True,
        "ym_start": ym_start,
        "ym_end": ym_end,
        "paths": result.get("paths", {}),
        "tasks": result.get("tasks", {}),
        "metadata": metadata,
    }

    debug_info = result.get("debug") or metadata.get("debug")
    if debug_info:
        response["debug"] = debug_info

    if request.export_target == "gcs":
        response["bucket"] = result.get("bucket")
        response["prefix"] = result.get("prefix")
    elif request.export_target == "drive":
        response["folder"] = result.get("folder")
        response["prefix"] = result.get("prefix")
    else:
        response["prefix"] = result.get("prefix")

    return response


