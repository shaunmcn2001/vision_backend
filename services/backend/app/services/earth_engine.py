# services/backend/app/services/earth_engine.py
"""
Earth Engine helpers:
- ensure_ee(): one-time initialization using ADC (service-account JSON mounted)
- ee_ping(): tiny query to validate EE access
- geometry_from_geojson(obj): safely build ee.Geometry from GeoJSON/Feature
- monthly_ndvi_series(geom_geojson, start, end): list of {date, ndvi_mean}
"""

from __future__ import annotations
import os
import threading
from typing import Dict, Any, List

import ee  # google-earth-engine Python API

# ---- one-time init guard -----------------------------------------------------
_init_lock = threading.Lock()
_initialized = False


def ensure_ee() -> None:
    """
    Initialize the Earth Engine client once per process.

    Relies on Application Default Credentials:
    - GOOGLE_APPLICATION_CREDENTIALS must point to a service-account JSON file,
      e.g. /secrets/ee-key.json (mounted via Cloud Run Secret volume).
    """
    global _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return

        # This will pick up GOOGLE_APPLICATION_CREDENTIALS automatically.
        ee.Initialize()  # raises if creds are missing/invalid
        _initialized = True


# ---- tiny diag ---------------------------------------------------------------
def ee_ping() -> Dict[str, Any]:
    """
    Lightweight request to confirm Earth Engine is available.
    Returns the size of a tiny one-day Sentinel-2 collection.
    """
    ensure_ee()
    col = ee.ImageCollection("COPERNICUS/S2_SR").filterDate("2020-01-01", "2020-01-02")
    return {"s2_sample_count": col.size().getInfo()}


# ---- helpers -----------------------------------------------------------------
def geometry_from_geojson(obj: Dict[str, Any]) -> ee.Geometry:
    """
    Accepts GeoJSON Polygon/MultiPolygon/Feature/FeatureCollection (first feature).
    Returns an ee.Geometry.
    """
    ensure_ee()

    # FeatureCollection → take the first feature
    if obj.get("type") == "FeatureCollection":
        feats = obj.get("features", [])
        if not feats:
            raise ValueError("FeatureCollection is empty.")
        obj = feats[0]

    # Feature → use its geometry
    if obj.get("type") == "Feature":
        geom = obj.get("geometry")
        if not geom:
            raise ValueError("Feature has no 'geometry'.")
        return ee.Geometry(geom)

    # Geometry object
    if "type" in obj and "coordinates" in obj:
        return ee.Geometry(obj)

    raise ValueError("Unsupported GeoJSON input. Provide Geometry/Feature/FeatureCollection.")


# ---- NDVI computation --------------------------------------------------------
def _with_ndvi(img: ee.Image) -> ee.Image:
    """Compute NDVI band and attach to the image."""
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    return img.addBands(ndvi)


def _basic_s2_sr_collection(geom: ee.Geometry, start: str, end: str) -> ee.ImageCollection:
    """
    Sentinel-2 Level-2A surface reflectance:
    - spatial filter
    - date range
    - cloudiness < 60%
    - (simple) mask snow/shadow using scene classification SCL != 9
    """
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR")
        .filterBounds(geom)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
        .map(lambda img: img.updateMask(img.select("SCL").neq(9)))
        .map(_with_ndvi)
    )
    return s2


def monthly_ndvi_series(geom_geojson: Dict[str, Any], start: str, end: str) -> List[Dict[str, Any]]:
    """
    Returns a list of {date: 'YYYY-MM', ndvi_mean: float} for each month in [start, end].

    Parameters
    ----------
    geom_geojson : dict
        GeoJSON Geometry/Feature/FeatureCollection (first feature used).
    start : str
        ISO date 'YYYY-MM-DD'
    end : str
        ISO date 'YYYY-MM-DD'
    """
    ensure_ee()
    geom = geometry_from_geojson(geom_geojson)
    s2 = _basic_s2_sr_collection(geom, start, end)

    # Build list of months
    start_d = ee.Date(start)
    end_d = ee.Date(end)
    month_count = end_d.difference(start_d, "month").floor().getInfo()

    out: List[Dict[str, Any]] = []
    for i in range(int(month_count) + 1):
        month_start = start_d.advance(i, "month")
        month_end = month_start.advance(1, "month")

        # Mean NDVI image for the month over the AOI
        ndvi_mean_img = s2.filterDate(month_start, month_end).select("NDVI").mean()

        # Reduce over the polygon
        stat = ndvi_mean_img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=10,
            bestEffort=True,
            maxPixels=1e13,
        )

        out.append(
            {
                "date": month_start.format("YYYY-MM").getInfo(),
                "ndvi_mean": stat.get("NDVI"),
            }
        )

    # Convert server-side objects to client values
    # (We already used getInfo() for date; NDVI may still be a computed object)
    cleaned = []
    for row in out:
        val = row["ndvi_mean"]
        if hasattr(val, "getInfo"):
            try:
                val = val.getInfo()
            except Exception:
                val = None
        cleaned.append({"date": row["date"], "ndvi_mean": val})

    return cleaned
