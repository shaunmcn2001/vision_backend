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
    method: Literal["ndvi_linear"] = Field(
        "ndvi_linear",
        description="NDVI → mean composite → fixed thresholds classification",
    )

    # 3 or 5 zones only
    n_classes: Literal[3, 5] = Field(
        5, description="Number of production zones (3 or 5)"
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
    mmu_ha: float = Field(zone_service.DEFAULT_MIN_MAPPING_UNIT_HA, gt=0)
    smooth_radius_m: float = Field(zone_service.DEFAULT_SMOOTH_RADIUS_M, ge=0)
    simplify_tol_m: float = Field(zone_service.DEFAULT_SIMPLIFY_TOL_M, ge=0)
    simplify_buffer_m: float = Field(zone_service.DEFAULT_SIMPLIFY_BUFFER_M)
    export_target: Literal["zip", "gcs", "drive", "local"] = Field(
        "zip", description="Destination for exports"
    )
    gcs_bucket: Optional[str] = Field(None, description="Override GCS bucket for exports")
    gcs_prefix: Optional[str] = Field(
        None, description="Optional prefix before zones/ when exporting to GCS"
    )
    include_zonal_stats: bool = Field(True, description="Export per-zone statistics CSV")

    # threshold controls
    ndvi_min: Optional[float] = Field(
        None, description="Lower bound for NDVI fixed-bin classification (default 0.35)"
    )
    ndvi_max: Optional[float] = Field(
        None, description="Upper bound for NDVI fixed-bin classification (default 0.73)"
    )
    custom_thresholds: Optional[List[float]] = Field(
        None,
        description=(
            "Optional explicit edges for bins (n_classes+1 values). "
            "If provided, ndvi_min/max are ignored."
        ),
    )

    @root_validator(pre=True)
    def _coerce_months(cls, values: dict) -> dict:
        months = values.get("months")
        start_month = values.get("start_month")
        end_month = values.get("end_month")
        start_date_value = values.get("start_date")
        end_date_value = values.get("end_date")

        modes_selected = sum(
            1 for provided in (
                bool(months),
                start_month is not None or end_month is not None,
                start_date_value is not None or end_date_value is not None
            ) if provided
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
            except ValueError as exc:
                raise ValueError(f"Invalid month format for {field_name}: {value}") from exc

        def _month_end(dt: datetime) -> date:
            import calendar as _cal
            last_day = _cal.monthrange(dt.year, dt.month)[1]
            return date(dt.year, dt.month, last_day)

        def _parse_date(value, field_name: str) -> date:
            if isinstance(value, date):
                return value
            value_str = str(value).strip()
            try:
                return date.fromisoformat(value_str)
            except ValueError as exc:
                raise ValueError(f"Invalid date format for {field_name}: {value}") from exc

        if months:
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

        if start_month is not None or end_month is not None:
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

        # Date range
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

    try:
        result = zone_service.export_selected_period_zones(
            request.aoi_geojson,
            request.aoi_name,
            request.months,
            start_date=request.start_date,
            end_date=request.end_date,
            cloud_prob_max=request.cloud_prob_max,
            n_classes=int(request.n_classes),
            mmu_ha=float(request.mmu_ha),
            smooth_radius_m=float(request.smooth_radius_m),
            simplify_tol_m=int(request.simplify_tol_m),
            simplify_buffer_m=int(request.simplify_buffer_m),
            export_target=request.export_target,
            gcs_bucket=resolved_bucket,
            gcs_prefix=request.gcs_prefix,
            include_zonal_stats=request.include_zonal_stats,
            method="ndvi_linear",
            ndvi_min=request.ndvi_min,
            ndvi_max=request.ndvi_max,
            custom_thresholds=request.custom_thresholds,
        )
    except ValueError as exc:
        logger.warning(
            "Zone production request validation failed for AOI %s (target %s): %s",
            request.aoi_name, request.export_target, exc, exc_info=True,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception(
            "Zone production runtime failure for AOI %s (target %s): %s",
            request.aoi_name, request.export_target, exc,
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    result.pop("artifacts", None)
    response = {
        "ok": True,
        "paths": result.get("paths", {}),
        "tasks": result.get("tasks", {}),
        "metadata": sanitize_for_json(result.get("metadata")),
        "palette": result.get("palette"),
        "thresholds": result.get("thresholds"),
        "prefix": result.get("prefix"),
        "bucket": result.get("bucket"),
        "folder": result.get("folder"),
    }
    return response
