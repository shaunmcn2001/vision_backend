from __future__ import annotations

import os
import calendar
import logging
from datetime import date, datetime
from typing import List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, root_validator, validator
from shapely.geometry import shape

from app.services import zones as zone_service
from app.utils.sanitization import sanitize_for_json


router = APIRouter(prefix="/zones", tags=["zones"])
logger = logging.getLogger(__name__)


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
    method: Literal["ndvi_percentiles", "ndvi_kmeans"] = Field(
        "ndvi_percentiles",
        description="Classification method for production zones",
    )
    mode: Literal["auto", "quantile", "linear"] = Field(
        "auto",
        description="Binning strategy for NDVI zones: 'auto' (adaptive), 'quantile', or 'linear'",
    )

    # optional fixed-range overrides for linear mode (and available to others if desired)
    min_ndvi: Optional[float] = Field(
        None, ge=-1.0, le=1.0, description="Lower NDVI bound for fixed-range zoning."
    )
    max_ndvi: Optional[float] = Field(
        None, ge=-1.0, le=1.0, description="Upper NDVI bound for fixed-range zoning."
    )

    months: Optional[List[str]] = Field(
        None, description="Months in YYYY-MM format"
    )
    start_month: Optional[str] = Field(
        None, description="Start month in YYYY-MM format (inclusive)",
    )
    end_month: Optional[str] = Field(
        None, description="End month in YYYY-MM format (inclusive)",
    )
    start_date: Optional[date] = Field(
        None, description="Start date in YYYY-MM-DD format (inclusive)",
    )
    end_date: Optional[date] = Field(
        None, description="End date in YYYY-MM-DD format (inclusive)",
    )
    cloud_prob_max: int = Field(zone_service.DEFAULT_CLOUD_PROB_MAX, ge=0, le=100)
    n_classes: int = Field(zone_service.DEFAULT_N_CLASSES, ge=3, le=7)
    cv_mask_threshold: float = Field(zone_service.DEFAULT_CV_THRESHOLD, ge=0)
    mmu_ha: float = Field(zone_service.DEFAULT_MIN_MAPPING_UNIT_HA, gt=0)
    smooth_radius_m: float = Field(zone_service.DEFAULT_SMOOTH_RADIUS_M, ge=0)
    open_radius_m: float = Field(zone_service.DEFAULT_OPEN_RADIUS_M, ge=0)
    close_radius_m: float = Field(zone_service.DEFAULT_CLOSE_RADIUS_M, ge=0)
    simplify_tol_m: float = Field(zone_service.DEFAULT_SIMPLIFY_TOL_M, ge=0)
    simplify_buffer_m: float = Field(zone_service.DEFAULT_SIMPLIFY_BUFFER_M)
    export_target: Literal["zip", "gcs", "drive"] = Field(
        "zip", description="Destination for exports"
    )
    gcs_bucket: Optional[str] = Field(None, description="Override GCS bucket for exports")
    gcs_prefix: Optional[str] = Field(
        None, description="Optional prefix before zones/ when exporting to GCS"
    )
    include_zonal_stats: bool = Field(True, description="Export per-zone statistics CSV")
    apply_stability_mask: Optional[bool] = Field(
        None,
        description=(
            "When false, skip the stability mask used to drop high-variance pixels before "
            "classifying zones.  When not provided, the service honours the APPLY_STABILITY "
            "environment variable."
        ),
    )

    @validator("max_ndvi")
    def _check_ndvi_bounds(cls, v, values):
        min_val = values.get("min_ndvi")
        if v is not None and min_val is not None and v <= min_val:
            raise ValueError("max_ndvi must be greater than min_ndvi")
        return v

    @root_validator(pre=True)
    def _coerce_months(cls, values: dict) -> dict:
        months = values.get("months")
        start_month = values.get("start_month")
        end_month = values.get("end_month")
        start_date_value = values.get("start_date")
        end_date_value = values.get("end_date")

        months_provided = bool(months)
        month_range_provided = start_month is not None or end_month is not None
        date_range_provided = start_date_value is not None or end_date_value is not None

        modes_selected = sum(
            1
            for provided in (months_provided, month_range_provided, date_range_provided)
            if provided
        )

        if modes_selected == 0:
            raise ValueError(
                "Either months[], start_month/end_month, or start_date/end_date must be provided"
            )
        if modes_selected > 1:
            raise ValueError(
                "Provide only one of months[], start_month/end_month, or start_date/end_date"
            )

        def _parse_month(value: str, field_name: str) -> datetime:
            month_str = str(value).strip()
            try:
                return datetime.strptime(month_str, "%Y-%m")
            except ValueError as exc:  # pragma: no cover - defensive
                raise ValueError(f"Invalid month format for {field_name}: {value}") from exc

        def _month_end(dt: datetime) -> date:
            last_day = calendar.monthrange(dt.year, dt.month)[1]
            return date(dt.year, dt.month, last_day)

        def _parse_date(value, field_name: str) -> date:
            if isinstance(value, date):
                return value
            value_str = str(value).strip()
            try:
                return date.fromisoformat(value_str)
            except ValueError as exc:
                raise ValueError(f"Invalid date format for {field_name}: {value}") from exc

        if months_provided:
            parsed_months: List[str] = []
            start_dt: datetime | None = None
            end_dt: datetime | None = None
            for raw in months:
                parsed = _parse_month(raw, "months[]")
                parsed_months.append(parsed.strftime("%Y-%m"))
                start_dt = parsed if start_dt is None or parsed < start_dt else start_dt
                end_dt = parsed if end_dt is None or parsed > end_dt else end_dt

            if not parsed_months:
                raise ValueError("At least one month must be provided")

            start_dt = start_dt or _parse_month(parsed_months[0], "months[]")
            end_dt = end_dt or start_dt
            values["months"] = parsed_months
            values["start_date"] = date(start_dt.year, start_dt.month, 1)
            values["end_date"] = _month_end(end_dt)
            return values

        if month_range_provided:
            if start_month is None or end_month is None:
                raise ValueError("Both start_month and end_month must be provided together")
            start_dt = _parse_month(start_month, "start_month")
            end_dt = _parse_month(end_month, "end_month")
            if end_dt < start_dt:
                raise ValueError("end_month must be on or after start_month")

            generated: List[str] = []
            cursor = start_dt
            while cursor <= end_dt:
                generated.append(cursor.strftime("%Y-%m"))
                if cursor.month == 12:
                    cursor = cursor.replace(year=cursor.year + 1, month=1)
                else:
                    cursor = cursor.replace(month=cursor.month + 1)

            values["months"] = generated
            values["start_date"] = date(start_dt.year, start_dt.month, 1)
            values["end_date"] = _month_end(end_dt)
            return values

        # Date range mode
        if start_date_value is None or end_date_value is None:
            raise ValueError("Both start_date and end_date must be provided together")

        start_dt = _parse_date(start_date_value, "start_date")
        end_dt = _parse_date(end_date_value, "end_date")
        if end_dt < start_dt:
            raise ValueError("end_date must be on or after start_date")

        generated: List[str] = []
        cursor = date(start_dt.year, start_dt.month, 1)
        end_cursor = date(end_dt.year, end_dt.month, 1)
        while cursor <= end_cursor:
            generated.append(cursor.strftime("%Y-%m"))
            if cursor.month == 12:
                cursor = cursor.replace(year=cursor.year + 1, month=1)
            else:
                cursor = cursor.replace(month=cursor.month + 1)

        values["months"] = generated
        values["start_date"] = start_dt
        values["end_date"] = end_dt
        return values

    @validator("months")
    def _validate_months(cls, value: Optional[List[str]]) -> List[str]:
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

    destination = request.export_target

    try:
        result = zone_service.export_selected_period_zones(
            request.aoi_geojson,
            months=request.months,
            aoi_name=request.aoi_name,
            destination=destination,
            method=request.method,
            start_date=request.start_date,
            end_date=request.end_date,
            cloud_prob_max=request.cloud_prob_max,
            n_classes=request.n_classes,
            cv_mask_threshold=request.cv_mask_threshold,
            min_mapping_unit_ha=request.mmu_ha,
            smooth_radius_m=request.smooth_radius_m,
            open_radius_m=request.open_radius_m,
            close_radius_m=request.close_radius_m,
            simplify_tolerance_m=request.simplify_tol_m,
            simplify_buffer_m=request.simplify_buffer_m,
            gcs_bucket=resolved_bucket,
            gcs_prefix=request.gcs_prefix,
            include_stats=request.include_zonal_stats,
            apply_stability_mask=request.apply_stability_mask,
            # NEW: pass mode and optional fixed NDVI bounds
            mode=request.mode,
            min_ndvi=request.min_ndvi,
            max_ndvi=request.max_ndvi,
        )
    except ValueError as exc:
        logger.warning(
            "Zone production request validation failed for AOI %s (target %s): %s",
            request.aoi_name,
            request.export_target,
            exc,
            exc_info=True,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception(
            "Zone production runtime failure for AOI %s (target %s): %s",
            request.aoi_name,
            request.export_target,
            exc,
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    result.pop("artifacts", None)
    metadata = result.get("metadata", {}) or {}
    used_months: List[str] = metadata.get("used_months") or request.months
    ym_start = used_months[0]
    ym_end = used_months[-1]

    palette = result.get("palette") or metadata.get("palette") if isinstance(metadata, dict) else None
    thresholds = result.get("thresholds") or (
        metadata.get("percentile_thresholds") if isinstance(metadata, dict) else None
    )

    response = {
        "ok": True,
        "ym_start": ym_start,
        "ym_end": ym_end,
        "paths": result.get("paths", {}),
        "tasks": result.get("tasks", {}),
        "metadata": metadata,
    }

    if palette is not None:
        response["palette"] = palette
    if thresholds is not None:
        response["thresholds"] = thresholds

    debug_info = result.get("debug") or metadata.get("debug")
    stability_meta = {}
    if isinstance(metadata, dict):
        stability_raw = metadata.get("stability")
        if isinstance(stability_raw, dict):
            stability_meta.update(stability_raw)
    if isinstance(debug_info, dict):
        stability_extra = debug_info.get("stability")
        if isinstance(stability_extra, dict):
            stability_meta.update({k: v for k, v in stability_extra.items() if v is not None})

    debug_payload = {
        "requested_months": request.months,
        "used_months": used_months,
        "skipped_months": metadata.get("skipped_months", []) if isinstance(metadata, dict) else [],
        "retry_thresholds": stability_meta.get("thresholds_tested", []),
        "stability": stability_meta or None,
    }

    # Keep any additional debug information provided by the service
    if isinstance(debug_info, dict):
        for key, value in debug_info.items():
            if key == "stability":
                continue
            debug_payload.setdefault(key, value)

    response["debug"] = debug_payload

    if request.export_target == "gcs":
        response["bucket"] = result.get("bucket")
        response["prefix"] = result.get("prefix")
    elif request.export_target == "drive":
        response["folder"] = result.get("folder")
        response["prefix"] = result.get("prefix")
    else:
        response["prefix"] = result.get("prefix")

    response["metadata"] = sanitize_for_json(response.get("metadata"))
    if "debug" in response:
        response["debug"] = sanitize_for_json(response.get("debug"))

    return response
