from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Any, Dict

import ee

from app import gee
from app.services.export import export_float_geotiff
from app.services.zones_logic import (
    build_mean_ndvi,
    classify_from_thresholds,
    thresholds_memory_safe,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.api.zones import ProductionZonesRequest

EXPORT_BASE = os.getenv("EXPORT_BASE_PATH", "/opt/exports")
_SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_-]+")


def _sanitize_name(name: str) -> str:
    cleaned = _SAFE_NAME_PATTERN.sub("_", name.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "aoi"


def _run_zones_production(req: "ProductionZonesRequest") -> Dict[str, Any]:
    gee.init_ee()
    aoi = ee.Geometry(req.aoi_geojson)
    start = req.start_date.isoformat() if req.start_date else None
    end = req.end_date.isoformat() if req.end_date else None
    if not start or not end:
        raise ValueError("start_date and end_date are required")

    mean_ndvi, n_imgs = build_mean_ndvi(aoi, start, end)
    thrs = thresholds_memory_safe(mean_ndvi, aoi, num_zones=req.n_classes)
    zones_raster = classify_from_thresholds(mean_ndvi, thrs)

    safe_name = _sanitize_name(req.aoi_name)
    out_dir = os.path.join(EXPORT_BASE, safe_name)

    ndvi_path = export_float_geotiff(
        mean_ndvi, aoi, f"{safe_name}_NDVI_mean", folder=out_dir
    )
    zones_path = export_float_geotiff(
        zones_raster.toFloat(), aoi, f"{safe_name}_ZONES", folder=out_dir
    )

    return {
        "n_images": n_imgs.getInfo(),
        "thresholds": thrs.getInfo(),
        "files": [ndvi_path, zones_path],
    }


__all__ = ["_run_zones_production"]
