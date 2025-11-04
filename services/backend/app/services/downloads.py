"""Helpers for generating direct Earth Engine download URLs."""

from __future__ import annotations

import json
from typing import Mapping

import ee

from app.config import get_settings
from app.services.earth_engine import to_geometry


class DownloadTooLargeError(ValueError):
    """Raised when an AOI exceeds the direct download size guard."""


def _prepare_region(aoi: Mapping[str, object], buffer_m: float = 0) -> ee.Geometry:
    geometry = to_geometry(aoi)
    if buffer_m:
        geometry = geometry.buffer(buffer_m)
    return geometry


def _check_max_pixels(geometry: ee.Geometry, scale: float) -> None:
    settings = get_settings()
    approx_pixels = geometry.area(maxError=1).divide(scale * scale)
    pixels = approx_pixels.getInfo()
    if pixels and pixels > settings.max_pixels_for_direct:
        raise DownloadTooLargeError(
            "AOI too large for direct download; reduce extent/date range."
        )


def image_geotiff_url(
    image: ee.Image,
    aoi: Mapping[str, object],
    *,
    name: str,
    scale: float = 10,
    crs: str | None = None,
    region_buffer_m: float = 0,
) -> str:
    """Construct a direct download URL for an image GeoTIFF."""
    geometry = _prepare_region(aoi, region_buffer_m)
    _check_max_pixels(geometry, scale)

    params: dict[str, object] = {
        "name": name,
        "region": json.dumps(geometry.getInfo()),
        "scale": scale,
        "filePerBand": False,
        "format": "GEO_TIFF",
    }
    if crs:
        params["crs"] = crs
    return image.getDownloadURL(params)


def table_shp_url(
    feature_collection: ee.FeatureCollection,
    *,
    name: str,
) -> str:
    """Return a direct download URL for a zipped shapefile."""
    try:
        return feature_collection.getDownloadURL(filetype="SHP", filename=name)
    except TypeError:
        params = {
            "format": "SHP",
            "filename": name,
        }
        return feature_collection.getDownloadURL(params)


def table_csv_url(
    feature_collection: ee.FeatureCollection,
    *,
    name: str,
) -> str:
    """Return a direct download URL for a CSV export."""
    try:
        return feature_collection.getDownloadURL(filetype="CSV", filename=name)
    except TypeError:
        params = {
            "format": "CSV",
            "filename": name,
        }
        return feature_collection.getDownloadURL(params)
