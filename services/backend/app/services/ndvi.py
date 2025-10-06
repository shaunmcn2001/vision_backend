from __future__ import annotations

import csv
import io
import json
import logging
import math
import os
import pathlib
from datetime import date
from typing import Any, Dict, Mapping, Optional

import ee
from ee import ServiceAccountCredentials

from app.services.gcs import (
    _bucket,
    download_json,
    exists,
    list_prefix,
    upload_json,
)
from app.services.image_stats import temporal_stats
from app.services.indices import normalize_index_code, resolve_index


DEFAULT_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
DEFAULT_SCALE = 10
SA_EMAIL = os.getenv(
    "EE_SERVICE_ACCOUNT", "ee-agri-worker@baradine-farm.iam.gserviceaccount.com"
)

logger = logging.getLogger(__name__)


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


EE_DEBUG_STATS = _env_flag("EE_DEBUG_STATS")
EE_MIN_VALID_PIXELS = max(int(os.getenv("EE_MIN_VALID_PIXELS", "100")), 0)
EE_MIN_VALID_PIXEL_RATIO = max(
    float(os.getenv("EE_MIN_VALID_PIXEL_RATIO", "0.01")), 0.0
)
EE_REDUCE_MAX_PIXELS = float(os.getenv("EE_REDUCE_MAX_PIXELS", "1e9"))


def _evaluate_server_value(value: Any) -> Any:
    if hasattr(value, "getInfo"):
        try:
            return value.getInfo()
        except Exception:  # pragma: no cover - defensive logging below
            return None
    return value


def _extract_reduce_value(result: Any, *, band_name: Optional[str] = None) -> Any:
    if isinstance(result, dict):
        if band_name and band_name in result and result[band_name] is not None:
            return result[band_name]
        for value in result.values():
            if value is not None:
                return value
        return None
    if band_name and hasattr(result, "get"):
        try:
            candidate = result.get(band_name)
        except Exception:  # pragma: no cover - defensive guard for EE objects
            candidate = None
        return _evaluate_server_value(candidate)
    return result


def _safe_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    candidate = _evaluate_server_value(value)
    if candidate is None:
        return None
    try:
        numeric = float(candidate)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _get_first_band_name(image: Any, fallback: Optional[str] = None) -> Optional[str]:
    bands = getattr(image, "bandNames", None)
    if not callable(bands):
        return fallback
    try:
        names = bands()
        names_value = _evaluate_server_value(names)
    except Exception:  # pragma: no cover - defensive guard for EE errors
        return fallback
    if isinstance(names_value, (list, tuple)) and names_value:
        return str(names_value[0])
    return fallback


def _reduce_scalar(
    image: Any,
    *,
    geometry: Any,
    scale: int,
    reducer: Any,
    label: str,
    band_name: Optional[str] = None,
) -> Optional[float]:
    if image is None or not hasattr(image, "reduceRegion"):
        return None
    try:
        reduced = image.reduceRegion(
            reducer=reducer,
            geometry=geometry,
            scale=scale,
            bestEffort=True,
            tileScale=4,
            maxPixels=EE_REDUCE_MAX_PIXELS,
        )
    except Exception as exc:  # pragma: no cover - Earth Engine failure
        if EE_DEBUG_STATS:
            logger.warning("EE reduceRegion failed for %s: %s", label, exc)
        return None

    try:
        reduced_info = _evaluate_server_value(reduced)
    except Exception as exc:  # pragma: no cover - defensive guard
        if EE_DEBUG_STATS:
            logger.warning("EE reduceRegion evaluation failed for %s: %s", label, exc)
        return None

    value = _extract_reduce_value(reduced_info, band_name=band_name)
    return _safe_number(value)


def _maybe_count_pixels(
    image: Any,
    *,
    geometry: Any,
    scale: int,
    band_hint: Optional[str],
    label: str,
) -> Optional[int]:
    band_name = _get_first_band_name(image, band_hint)
    count = _reduce_scalar(
        image,
        geometry=geometry,
        scale=scale,
        reducer=ee.Reducer.count(),
        label=label,
        band_name=band_name,
    )
    if count is None:
        return None
    return int(count)


def _maybe_log_collection_diagnostics(
    collection: ee.ImageCollection,
    *,
    geometry: ee.Geometry,
    band_name: str,
    scale: int,
):
    if not EE_DEBUG_STATS:
        return

    debug_logger = logger.getChild("ee")
    debug_logger.info(
        "EE debug [%s]: starting diagnostics for collection", band_name
    )

    try:
        first = ee.Image(collection.first())
        names = _evaluate_server_value(first.bandNames())
        debug_logger.info("EE debug [%s]: first image band names: %s", band_name, names)
    except Exception as exc:  # pragma: no cover - diagnostic only
        debug_logger.warning(
            "EE debug [%s]: failed to fetch first image band names: %s", band_name, exc
        )

    try:
        size = int(ee.Number(collection.size()).getInfo() or 0)
        debug_logger.info("EE debug [%s]: collection size: %s", band_name, size)
    except Exception as exc:  # pragma: no cover - diagnostic only
        debug_logger.warning(
            "EE debug [%s]: failed to evaluate collection size: %s", band_name, exc
        )

    def _img_minmax(img):  # pragma: no cover - EE server-side mapping
        image = ee.Image(img).select([band_name])
        mm = image.reduceRegion(
            reducer=ee.Reducer.minMax(),
            geometry=geometry,
            scale=scale,
            bestEffort=True,
            tileScale=4,
            maxPixels=EE_REDUCE_MAX_PIXELS,
        )
        return ee.Dictionary({
            "id": image.get("system:id"),
            "minmax": mm,
        })

    try:
        sample = ee.List(
            ee.ImageCollection(collection).toList(10).map(_img_minmax)
        ).getInfo()
        debug_logger.info(
            "EE debug [%s]: sample per-image min/max (first 10): %s",
            band_name,
            sample,
        )
    except Exception as exc:  # pragma: no cover - diagnostic only
        debug_logger.warning(
            "EE debug [%s]: failed to compute per-image min/max: %s",
            band_name,
            exc,
        )

    band_collection = ee.ImageCollection(collection).select([band_name])
    raw_sum = band_collection.reduce(ee.Reducer.sum()).rename(f"{band_name}_sum")
    raw_count = band_collection.reduce(ee.Reducer.count()).rename(
        f"{band_name}_count"
    )

    try:
        sum_stats = raw_sum.reduceRegion(
            reducer=ee.Reducer.minMax(),
            geometry=geometry,
            scale=scale,
            bestEffort=True,
            tileScale=4,
            maxPixels=EE_REDUCE_MAX_PIXELS,
        ).getInfo()
        debug_logger.info(
            "EE debug [%s]: raw_sum min/max: %s", band_name, sum_stats
        )
    except Exception as exc:  # pragma: no cover - diagnostic only
        debug_logger.warning(
            "EE debug [%s]: failed to compute raw_sum stats: %s", band_name, exc
        )

    try:
        count_stats = raw_count.reduceRegion(
            reducer=ee.Reducer.minMax(),
            geometry=geometry,
            scale=scale,
            bestEffort=True,
            tileScale=4,
            maxPixels=EE_REDUCE_MAX_PIXELS,
        ).getInfo()
        debug_logger.info(
            "EE debug [%s]: raw_count min/max: %s", band_name, count_stats
        )
    except Exception as exc:  # pragma: no cover - diagnostic only
        debug_logger.warning(
            "EE debug [%s]: failed to compute raw_count stats: %s", band_name, exc
        )

    safe_count = raw_count.where(raw_count.eq(0), 1)
    mean = raw_sum.divide(safe_count).rename(f"{band_name}_mean").updateMask(
        raw_count.gt(0)
    )

    try:
        mean_stats = mean.reduceRegion(
            reducer=ee.Reducer.minMax(),
            geometry=geometry,
            scale=scale,
            bestEffort=True,
            tileScale=4,
            maxPixels=EE_REDUCE_MAX_PIXELS,
        ).getInfo()
        debug_logger.info(
            "EE debug [%s]: mean min/max: %s", band_name, mean_stats
        )
    except Exception as exc:  # pragma: no cover - diagnostic only
        debug_logger.warning(
            "EE debug [%s]: failed to compute mean min/max: %s", band_name, exc
        )

    try:
        histogram = mean.reduceRegion(
            reducer=ee.Reducer.histogram(),
            geometry=geometry,
            scale=scale,
            bestEffort=True,
            tileScale=4,
            maxPixels=EE_REDUCE_MAX_PIXELS,
        ).getInfo()
        debug_logger.info(
            "EE debug [%s]: mean histogram: %s", band_name, histogram
        )
    except Exception as exc:  # pragma: no cover - diagnostic only
        debug_logger.warning(
            "EE debug [%s]: failed to compute mean histogram: %s",
            band_name,
            exc,
        )

    try:
        valid_pixels = mean.mask().reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=geometry,
            scale=scale,
            bestEffort=True,
            tileScale=4,
            maxPixels=EE_REDUCE_MAX_PIXELS,
        ).getInfo()
        total_pixels = mean.unmask().reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=geometry,
            scale=scale,
            bestEffort=True,
            tileScale=4,
            maxPixels=EE_REDUCE_MAX_PIXELS,
        ).getInfo()
        debug_logger.info(
            "EE debug [%s]: valid_pixels=%s total_pixels=%s",
            band_name,
            valid_pixels,
            total_pixels,
        )
    except Exception as exc:  # pragma: no cover - diagnostic only
        debug_logger.warning(
            "EE debug [%s]: failed to compute pixel counts: %s", band_name, exc
        )


def _log_fallback(
    *,
    code: str,
    month: int,
    reason: str,
    valid_pixels: Optional[int],
    total_pixels: Optional[int],
):
    extra = {"valid_pixels": valid_pixels, "total_pixels": total_pixels}
    logger.info(
        "EE fallback [%s] month=%02d reason=%s extras=%s",
        code,
        month,
        reason,
        {k: v for k, v in extra.items() if v is not None},
    )


def _find_or_write_keyfile() -> str:
    p = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if p and os.path.exists(p):
        return p
    for candidate in (
        "/etc/secrets/ee-key.json",
        "/etc/secrets/google-credentials.json",
        "/opt/render/project/src/ee-key.json",
    ):
        if os.path.exists(candidate):
            return candidate
    key_json = os.getenv("EE_KEY_JSON")
    if key_json:
        tmp = "/tmp/ee-key.json"
        pathlib.Path(tmp).write_text(json.dumps(json.loads(key_json)))
        return tmp
    raise RuntimeError(
        "No EE credentials. Set GOOGLE_APPLICATION_CREDENTIALS (file) or EE_KEY_JSON (env)."
    )


def init_ee():
    keyfile = _find_or_write_keyfile()
    creds = ServiceAccountCredentials(SA_EMAIL, keyfile)
    ee.Initialize(credentials=creds, opt_url="https://earthengine.googleapis.com")


def compute_monthly_index(
    geometry: dict,
    *,
    start: date,
    end: date,
    index_code: str,
    collection: str = DEFAULT_COLLECTION,
    scale: int = DEFAULT_SCALE,
    parameters: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    if end < start:
        raise ValueError("End date must be on or after start date.")

    init_ee()

    geom = ee.Geometry(geometry)
    definition, resolved_params = resolve_index(index_code, parameters)

    mapped_collection = (
        ee.ImageCollection(collection)
        .filterBounds(geom)
        .filterDate(start.isoformat(), end.isoformat())
        .map(lambda img: img.addBands(definition.compute(img, resolved_params)))
    )

    ndvi_collection = mapped_collection.select(definition.band_name)
    _maybe_log_collection_diagnostics(
        ndvi_collection,
        geometry=geom,
        band_name=definition.band_name,
        scale=scale,
    )

    values = []
    for month in _iterate_months(start, end):
        monthly_collection = mapped_collection.filter(
            ee.Filter.calendarRange(month, month, "month")
        )
        stats = temporal_stats(
            monthly_collection,
            band_name=definition.band_name,
            rename_prefix=definition.band_name,
            mean_band_name=definition.band_name,
        )

        mean_image = stats.get("mean")
        mean_band_name = _get_first_band_name(mean_image, definition.band_name)
        if (
            mean_image is not None
            and mean_band_name is not None
            and hasattr(mean_image, "select")
        ):
            mean_image = mean_image.select(mean_band_name)
        mean_value = _reduce_scalar(
            mean_image,
            geometry=geom,
            scale=scale,
            reducer=ee.Reducer.mean(),
            label=f"{definition.code} mean month {month}",
            band_name=mean_band_name,
        )

        valid_mask_image = stats.get("valid_mask")
        valid_pixels = _maybe_count_pixels(
            valid_mask_image,
            geometry=geom,
            scale=scale,
            band_hint=mean_band_name,
            label=f"{definition.code} valid pixels month {month}",
        )

        total_pixels: Optional[int] = None
        if valid_mask_image is not None and hasattr(valid_mask_image, "unmask"):
            total_pixels = _maybe_count_pixels(
                valid_mask_image.unmask(1),
                geometry=geom,
                scale=scale,
                band_hint=_get_first_band_name(valid_mask_image, mean_band_name),
                label=f"{definition.code} total pixels month {month}",
            )

        fallback_reasons: list[str] = []
        if mean_value is None:
            fallback_reasons.append("mean masked")
        if valid_pixels is not None and valid_pixels < EE_MIN_VALID_PIXELS:
            fallback_reasons.append(f"valid_pixels<{EE_MIN_VALID_PIXELS}")
        if (
            valid_pixels is not None
            and total_pixels
            and total_pixels > 0
            and EE_MIN_VALID_PIXEL_RATIO > 0
            and (valid_pixels / total_pixels) < EE_MIN_VALID_PIXEL_RATIO
        ):
            fallback_reasons.append(
                f"valid_ratio<{EE_MIN_VALID_PIXEL_RATIO:.3f}"
            )

        if fallback_reasons:
            fallback_image = stats.get("mean_unmasked")
            fallback_source = fallback_image
            if (
                fallback_image is not None
                and mean_band_name is not None
                and hasattr(fallback_image, "select")
            ):
                fallback_source = fallback_image.select(mean_band_name)
            fallback_value = _reduce_scalar(
                fallback_source,
                geometry=geom,
                scale=scale,
                reducer=ee.Reducer.mean(),
                label=f"{definition.code} mean_unmasked month {month}",
                band_name=mean_band_name,
            )
            if fallback_value is not None:
                mean_value = fallback_value
                _log_fallback(
                    code=definition.code,
                    month=int(month),
                    reason=", ".join(fallback_reasons),
                    valid_pixels=valid_pixels,
                    total_pixels=total_pixels,
                )
            elif EE_DEBUG_STATS:
                logger.warning(
                    "EE fallback [%s] month=%02d failed reasons=%s",
                    definition.code,
                    int(month),
                    ", ".join(fallback_reasons),
                )

        if EE_DEBUG_STATS and mean_value is not None:
            logger.debug(
                "EE diagnostics [%s] month=%02d mean=%.6f valid_pixels=%s total_pixels=%s",
                definition.code,
                int(month),
                mean_value,
                valid_pixels,
                total_pixels,
            )

        if mean_value is None:
            continue

        values.append({"month": int(month), definition.code: mean_value})

    return {
        "index": {
            "code": definition.code,
            "band": definition.band_name,
            "valid_range": list(definition.valid_range)
            if definition.valid_range is not None
            else None,
            "parameters": dict(resolved_params),
        },
        "data": values,
    }


def compute_monthly_index_for_year(
    geometry: dict,
    year: int,
    *,
    index_code: str,
    collection: str = DEFAULT_COLLECTION,
    scale: int = DEFAULT_SCALE,
    parameters: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    start = date(year, 1, 1)
    end = date(year, 12, 31)
    return compute_monthly_index(
        geometry,
        start=start,
        end=end,
        index_code=index_code,
        collection=collection,
        scale=scale,
        parameters=parameters,
    )


def gcs_index_prefix(index_code: str, field_id: str) -> str:
    normalized = normalize_index_code(index_code)
    return f"index-results/{normalized}/{field_id}"


def gcs_index_path(index_code: str, field_id: str, year: int) -> str:
    return f"{gcs_index_prefix(index_code, field_id)}/{year}.json"


def gcs_index_csv_path(index_code: str, field_id: str, year: int) -> str:
    return f"{gcs_index_prefix(index_code, field_id)}/{year}.csv"


def get_or_compute_and_cache_index(
    field_id: str,
    geometry: dict,
    year: int,
    *,
    index_code: str,
    collection: str = DEFAULT_COLLECTION,
    scale: int = DEFAULT_SCALE,
    parameters: Mapping[str, Any] | None = None,
    force: bool = False,
) -> Dict[str, Any]:
    path = gcs_index_path(index_code, field_id, year)
    if not force and exists(path):
        return download_json(path)

    result = compute_monthly_index_for_year(
        geometry,
        year,
        index_code=index_code,
        collection=collection,
        scale=scale,
        parameters=parameters,
    )

    payload = {
        "field_id": field_id,
        "year": year,
        "index": result["index"],
        "data": result["data"],
    }
    upload_json(payload, path)

    csv_path = gcs_index_csv_path(index_code, field_id, year)
    upload_index_csv(result["data"], csv_path, result["index"]["code"])

    return payload


def upload_index_csv(rows: list[dict], path: str, index_key: str):
    bucket = _bucket()
    blob = bucket.blob(path)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["month", index_key])
    writer.writeheader()
    writer.writerows(rows)
    blob.cache_control = "no-cache"
    blob.upload_from_string(buffer.getvalue(), content_type="text/csv")


def list_cached_years(field_id: str, index_code: str) -> list[int]:
    prefix = gcs_index_prefix(index_code, field_id)
    names = list_prefix(f"{prefix}/")
    years: set[int] = set()
    for name in names:
        fname = name.split("/")[-1]
        if fname.endswith(".json"):
            try:
                years.add(int(fname[:-5]))
            except ValueError:
                continue
    return sorted(years)


def _iterate_months(start: date, end: date) -> list[int]:
    months: list[int] = []
    year = start.year
    month = start.month
    while (year, month) <= (end.year, end.month):
        months.append(month)
        month += 1
        if month > 12:
            month = 1
            year += 1
    return months


__all__ = [
    "DEFAULT_COLLECTION",
    "DEFAULT_SCALE",
    "compute_monthly_index",
    "compute_monthly_index_for_year",
    "get_or_compute_and_cache_index",
    "gcs_index_csv_path",
    "gcs_index_path",
    "gcs_index_prefix",
    "init_ee",
    "list_cached_years",
    "upload_index_csv",
]
