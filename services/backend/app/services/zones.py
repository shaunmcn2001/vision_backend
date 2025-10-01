"""Production zone workflow built on Sentinel-2 monthly composites."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import calendar
import os
import math
from typing import Dict, List, Mapping, Sequence, Tuple, Union

import ee

from app import gee
from app.exports import sanitize_name
from app.utils.geometry import area_ha


DEFAULT_CLOUD_PROB_MAX = 40
DEFAULT_N_CLASSES = 5
DEFAULT_CV_THRESHOLD = 0.5
DEFAULT_MIN_MAPPING_UNIT_HA = 1.5
DEFAULT_SMOOTH_RADIUS_M = 30
DEFAULT_OPEN_RADIUS_M = 10
DEFAULT_CLOSE_RADIUS_M = 10
DEFAULT_SIMPLIFY_TOL_M = 5
DEFAULT_SIMPLIFY_BUFFER_M = 3
DEFAULT_METHOD = "ndvi_percentiles"
DEFAULT_SAMPLE_SIZE = 8000
DEFAULT_SCALE = 10
# IMPORTANT: processing uses the native S2 projection (meters).
# Exports use a metric CRS so scale=10 means 10 meters.
DEFAULT_EXPORT_CRS = "EPSG:3857"


def _parse_bool_env(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    trimmed = value.strip().lower()
    if trimmed in {"", "none"}:
        return default
    if trimmed in {"0", "false", "no", "off"}:
        return False
    if trimmed in {"1", "true", "yes", "on"}:
        return True
    return default


APPLY_STABILITY = _parse_bool_env(os.getenv("APPLY_STABILITY"), True)


def set_apply_stability(enabled: bool | None) -> None:
    """Temporarily override the APPLY_STABILITY flag.

    Passing ``None`` re-evaluates the environment variable.
    """
    global APPLY_STABILITY
    if enabled is None:
        APPLY_STABILITY = _parse_bool_env(os.getenv("APPLY_STABILITY"), True)
    else:
        APPLY_STABILITY = bool(enabled)


ZONE_PALETTE: tuple[str, ...] = (
    "#112f1d",
    "#1b4d2a",
    "#2c6a39",
    "#3f8749",
    "#58a35d",
    "#80bf7d",
    "#b6dcb1",
)

STABILITY_THRESHOLD_SEQUENCE = [0.5, 1.0, 1.5, 2.0]
MIN_STABILITY_SURVIVAL_RATIO = 0.0

NDVI_PERCENTILE_MIN = 0.0
NDVI_PERCENTILE_MAX = 0.6

STABILITY_MASK_EMPTY_ERROR = (
    "All pixels were masked out by the stability threshold when computing NDVI percentiles. "
    "Try lowering the coefficient of variation threshold, expanding the selected months, "
    "or switching the zone method."
)


@dataclass(frozen=True)
class ZoneArtifacts:
    """Container for the images/vectors used for production zone exports."""

    zone_image: ee.Image
    zone_vectors: ee.FeatureCollection
    zonal_stats: ee.FeatureCollection | None
    geometry: ee.Geometry


def _ordered_months(months: Sequence[str]) -> List[str]:
    unique: dict[str, datetime] = {}
    for raw in months:
        month_str = str(raw).strip()
        if month_str in unique:
            continue
        try:
            parsed = datetime.strptime(month_str, "%Y-%m")
        except ValueError as exc:
            raise ValueError(f"Invalid month format: {raw}") from exc
        unique[month_str] = parsed

    ordered = sorted(unique.items(), key=lambda item: item[1])
    return [month for month, _ in ordered]


def _month_range_dates(months: Sequence[str]) -> tuple[date, date]:
    ordered = _ordered_months(months)
    if not ordered:
        raise ValueError("At least one month must be supplied")
    start_dt = datetime.strptime(ordered[0], "%Y-%m")
    end_dt = datetime.strptime(ordered[-1], "%Y-%m")
    start_day = date(start_dt.year, start_dt.month, 1)
    end_last_day = calendar.monthrange(end_dt.year, end_dt.month)[1]
    end_day = date(end_dt.year, end_dt.month, end_last_day)
    return start_day, end_day


def _months_from_dates(start_date: date, end_date: date) -> List[str]:
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")
    months: List[str] = []
    cursor = date(start_date.year, start_date.month, 1)
    end_cursor = date(end_date.year, end_date.month, 1)
    while cursor <= end_cursor:
        months.append(cursor.strftime("%Y-%m"))
        if cursor.month == 12:
            cursor = cursor.replace(year=cursor.year + 1, month=1)
        else:
            cursor = cursor.replace(month=cursor.month + 1)
    return months


def _month_bounds(months: Sequence[str]) -> tuple[str, str]:
    if not months:
        raise ValueError("At least one month must be supplied")
    ordered = _ordered_months(months)
    return ordered[0], ordered[-1]


def _export_prefix(aoi_name: str, months: Sequence[str]) -> str:
    start, end = _month_bounds(months)
    safe_name = sanitize_name(aoi_name or "aoi")
    return f"zones/PROD_{start.replace('-', '')}_{end.replace('-', '')}_{safe_name}_zones"


def export_prefix(aoi_name: str, months: Sequence[str]) -> str:
    """Public helper that returns the export prefix for a zone export run."""
    return _export_prefix(aoi_name, months)


def month_bounds(months: Sequence[str]) -> tuple[str, str]:
    """Return the first and last months used in an export request."""
    return _month_bounds(months)


def resolve_export_bucket(explicit: str | None = None) -> str:
    bucket = (explicit or os.getenv("GEE_GCS_BUCKET") or os.getenv("GCS_BUCKET") or "").strip()
    if not bucket:
        raise RuntimeError("GEE_GCS_BUCKET or GCS_BUCKET must be set for zone exports")
    return bucket


def _resolve_geometry(aoi: Union[dict, ee.Geometry]) -> ee.Geometry:
    if isinstance(aoi, ee.Geometry):
        return aoi
    return gee.geometry_from_geojson(aoi)


def _attach_cloud_probability(
    collection: ee.ImageCollection, probability: ee.ImageCollection
) -> ee.ImageCollection:
    join = ee.Join.saveFirst("cloud_prob")
    matches = join.apply(
        primary=collection,
        secondary=probability,
        condition=ee.Filter.equals(leftField="system:index", rightField="system:index"),
    )

    def _add_probability(image: ee.Image) -> ee.Image:
        cloud_match = image.get("cloud_prob")
        probability_band = ee.Image(
            ee.Algorithms.If(
                ee.Algorithms.IsEqual(cloud_match, None),
                ee.Image.constant(0),
                ee.Image(cloud_match).select("probability"),
            )
        ).rename("cloud_probability")
        return image.addBands(probability_band)

    return ee.ImageCollection(matches).map(_add_probability)


def _mask_sentinel2_scene(image: ee.Image, cloud_prob_max: int) -> ee.Image:
    qa = image.select("QA60")
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    qa_mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))

    prob_mask = image.select("cloud_probability").lte(cloud_prob_max)

    scl = image.select("SCL")
    shadow_mask = scl.neq(3).And(scl.neq(11))

    combined_mask = qa_mask.And(prob_mask).And(shadow_mask)
    scaled = image.updateMask(combined_mask).divide(10_000)
    selected = scaled.select(list(gee.S2_BANDS))
    return selected.copyProperties(image, ["system:time_start"])


def _native_reproject(img: ee.Image) -> ee.Image:
    """Reproject to native Sentinel-2 10 m projection (meters)."""
    # Use B8 (10 m) projection as the reference
    proj = img
