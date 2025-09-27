from __future__ import annotations

import os
from datetime import datetime
from typing import List, Optional

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
    method: str = Field(zone_service.DEFAULT_METHOD, description="Zone generation method")
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


class ProductionZones5YRequest(_BaseAOIRequest):
    years_back: int = Field(5, gt=0)
    growth_months: Optional[List[str]] = Field(
        None,
        description="Optional list of growth months (MM) to restrict NDVI sampling",
    )
    cloud_prob_max: int = Field(zone_service.DEFAULT_CLOUD_PROB_MAX, ge=0, le=100)
    n_classes: int = Field(zone_service.DEFAULT_N_CLASSES, ge=3, le=7)
    cv_mask_threshold: float = Field(zone_service.DEFAULT_CV_THRESHOLD, ge=0)
    mmu_ha: float = Field(zone_service.DEFAULT_MIN_MAPPING_UNIT_HA, gt=0)
    smooth_kernel_px: int = Field(zone_service.DEFAULT_SMOOTH_KERNEL_PX, ge=0)
    simplify_tol_m: float = Field(zone_service.DEFAULT_SIMPLIFY_TOL_M, ge=0)
    method: str = Field(zone_service.DEFAULT_METHOD, description="Zone generation method")
    dem_asset: Optional[str] = Field(None, description="Optional DEM asset ID")
    include_zonal_stats: bool = Field(True, description="Export per-zone statistics CSV")
    export_target: str = Field("gcs", description="Destination: gcs, drive, or zip")
    gcs_bucket: Optional[str] = Field(None, description="Target GCS bucket")
    gcs_prefix: Optional[str] = Field(None, description="Optional GCS prefix before zones/")
    drive_folder: Optional[str] = Field(None, description="Override Drive folder name")

    @validator("growth_months", pre=True)
    def _validate_growth_months(cls, value):
        if value in (None, "", []):
            return None
        if isinstance(value, str):
            value = [value]
        months: List[str] = []
        for item in value:
            month_str = str(item).strip()
            if len(month_str) != 2 or not month_str.isdigit():
                raise ValueError("growth_months entries must be MM strings")
            month_int = int(month_str)
            if month_int < 1 or month_int > 12:
                raise ValueError("growth_months entries must be between 01 and 12")
            if month_str not in months:
                months.append(month_str)
        return months

    @validator("method")
    def _normalize_method(cls, value: str) -> str:
        method_key = value.strip().lower()
        if method_key not in {"ndvi_percentiles", "multiindex_kmeans"}:
            raise ValueError("Unsupported method for production zones")
        return method_key

    @validator("export_target")
    def _validate_target(cls, value: str) -> str:
        target = value.strip().lower()
        if target not in {"gcs", "drive", "zip"}:
            raise ValueError("export_target must be one of gcs, drive, or zip")
        return target

@router.post("/production")
def create_production_zones(request: ProductionZonesRequest):
    try:
        artifacts = zone_service.build_zone_artifacts(
            request.aoi_geojson,
            months=request.months,
            cloud_prob_max=request.cloud_prob_max,
            cv_mask_threshold=request.cv_mask_threshold,
            n_classes=request.n_classes,
            min_mapping_unit_ha=request.mmu_ha,
            smooth_kernel_px=request.smooth_kernel_px,
            simplify_tolerance_m=request.simplify_tol_m,
            method=request.method,
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

    prefix = zone_service.export_prefix(request.aoi_name, request.months)
    stats_prefix = prefix + "_zonal_stats"
    start_month, end_month = zone_service.month_bounds(request.months)

    paths = {
        "raster": f"gs://{bucket}/{prefix}.tif",
        "vectors": f"gs://{bucket}/{prefix}.shp",
        "zonal_stats": (
            f"gs://{bucket}/{stats_prefix}.csv" if request.include_zonal_stats else None
        ),
    }

    tasks = zone_service.start_zone_exports(
        artifacts,
        aoi_name=request.aoi_name,
        months=request.months,
        bucket=bucket,
        include_stats=request.include_zonal_stats,
    )

    def _task_payload(name: str, task):
        if task is None:
            return None
        try:
            status = task.status() or {}
        except Exception:  # pragma: no cover - defensive
            status = {}

        destination = paths.get(name)
        destination_uris = status.get("destination_uris")
        if not destination_uris and destination:
            destination_uris = [destination]

        payload = {
            "id": getattr(task, "id", None),
            "state": status.get("state"),
            "type": status.get("type"),
            "destination_uris": destination_uris,
        }
        if destination:
            payload["destination_uri"] = destination
        error = status.get("error_message") or status.get("error_details")
        if error:
            payload["error"] = error
        return payload

    return {
        "bucket": bucket,
        "paths": paths,
        "tasks": {
            "raster": _task_payload("raster", tasks["raster"]),
            "vectors": _task_payload("vectors", tasks["vectors"]),
            "zonal_stats": _task_payload("zonal_stats", tasks["stats"]),
        },
        "metadata": {
            "aoi_name": request.aoi_name,
            "months": request.months,
            "month_start": start_month,
            "month_end": end_month,
            "n_classes": request.n_classes,
            "method": request.method,
        },
    }


@router.post("/production5y")
def create_production_zones_5y(request: ProductionZones5YRequest):
    try:
        artifacts, window = zone_service.build_production5y_zone_artifacts(
            request.aoi_geojson,
            years_back=request.years_back,
            growth_months=request.growth_months,
            cloud_prob_max=request.cloud_prob_max,
            n_classes=request.n_classes,
            cv_mask_threshold=request.cv_mask_threshold,
            min_mapping_unit_ha=request.mmu_ha,
            smooth_kernel_px=request.smooth_kernel_px,
            simplify_tolerance_m=request.simplify_tol_m,
            method=request.method,
            dem_asset=request.dem_asset,
            include_stats=request.include_zonal_stats,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    prefix_base = zone_service.production5y_export_prefix(
        request.aoi_name, window.start_month, window.end_month
    )

    target = request.export_target
    tasks: dict = {}
    paths: dict = {}
    extra: dict = {}

    if target == "gcs":
        try:
            bucket = zone_service.resolve_export_bucket(request.gcs_bucket)
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        prefix = prefix_base
        if request.gcs_prefix:
            cleaned = request.gcs_prefix.strip().strip("/")
            if cleaned:
                prefix = f"{cleaned}/{prefix_base}"

        tasks = zone_service.start_zone_exports(
            artifacts,
            aoi_name=request.aoi_name,
            months=window.months,
            bucket=bucket,
            include_stats=request.include_zonal_stats,
            prefix_override=prefix,
        )

        stats_prefix = prefix + "_zonal_stats"
        paths = {
            "raster": f"gs://{bucket}/{prefix}.tif",
            "vectors": f"gs://{bucket}/{prefix}.shp",
            "zonal_stats": (
                f"gs://{bucket}/{stats_prefix}.csv" if request.include_zonal_stats else None
            ),
        }
        extra = {"bucket": bucket, "prefix": prefix}
    elif target == "drive":
        folder = (request.drive_folder or os.getenv("GEE_DRIVE_FOLDER") or "Sentinel2_Indices").strip()
        if not folder:
            folder = "Sentinel2_Indices"
        folder = folder.rstrip("/")
        if not folder.endswith("zones"):
            folder = f"{folder}/zones"

        drive_prefix = prefix_base.split("/")[-1]
        tasks = zone_service.start_zone_exports_drive(
            artifacts,
            folder=folder,
            prefix=drive_prefix,
            include_stats=request.include_zonal_stats,
        )
        paths = {
            "raster": f"drive://{folder}/{drive_prefix}.tif",
            "vectors": f"drive://{folder}/{drive_prefix}.shp",
            "zonal_stats": (
                f"drive://{folder}/{drive_prefix}_zonal_stats.csv"
                if request.include_zonal_stats
                else None
            ),
        }
        extra = {"folder": folder, "file_prefix": drive_prefix}
    else:
        raise HTTPException(status_code=400, detail="zip export_target is not yet supported")

    def _task_payload(task):
        if task is None:
            return None
        return {"id": getattr(task, "id", None)}

    return {
        "target": target,
        **extra,
        "paths": paths,
        "tasks": {
            "raster": _task_payload(tasks.get("raster")),
            "vectors": _task_payload(tasks.get("vectors")),
            "zonal_stats": _task_payload(tasks.get("stats")),
        },
        "metadata": {
            "aoi_name": request.aoi_name,
            "years_back": request.years_back,
            "growth_months": request.growth_months or [],
            "month_start": window.start_month,
            "month_end": window.end_month,
            "months": window.months,
            "n_classes": request.n_classes,
            "method": request.method,
            "dem_asset": request.dem_asset,
        },
    }

