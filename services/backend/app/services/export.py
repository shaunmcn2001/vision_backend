from __future__ import annotations

import os
from typing import Optional

import ee

from app.services.io_utils import stream_to_file

EE_MAXPIXELS = float(os.getenv("EE_MAXPIXELS", "10000000000000"))


def export_float_geotiff(
    image: ee.Image, aoi: ee.Geometry, name: str, folder: str = "/opt/exports"
) -> str:
    image = image.toFloat().clip(aoi)
    try:
        projection = image.projection()
        crs: Optional[str] = projection.crs().getInfo()
    except Exception:
        crs = "EPSG:4326"
    url = image.getDownloadURL(
        {
            "name": name,
            "scale": 10,
            "crs": crs,
            "region": aoi,
            "format": "GeoTIFF",
            "maxPixels": EE_MAXPIXELS,
        }
    )
    out_path = os.path.join(folder, f"{name}.tif")
    stream_to_file(url, out_path, timeout=600)
    return out_path


__all__ = ["export_float_geotiff"]
