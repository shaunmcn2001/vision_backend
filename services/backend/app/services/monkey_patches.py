# services/backend/app/services/monkey_patches.py
"""
Exporter patch v5 – use the image's native projection for export.

Why this fixes "one value":
- Some grids (e.g., Web Mercator) + AOI/scale combos can collapse sampling.
- Exporting in the image's own UTM CRS at its nominal scale preserves per-pixel variation.
- We do NOT call .reproject() (keeps masks/values intact).
"""

from __future__ import annotations
import logging
import os
import shutil
from pathlib import Path
from typing import Optional
from urllib.request import urlopen

import ee

from app.services import zones as _zones  # the live module we patch

log = logging.getLogger(__name__)

# Keep original in case we want to restore.
_ORIG_download_image_to_path = getattr(_zones, "_download_image_to_path", None)

def _proj_info(image: ee.Image) -> tuple[str, int]:
    """Return (crs, nominal_scale_m) from image.projection(); robust fallbacks."""
    crs = "EPSG:3857"
    scale = 10
    try:
        info = image.projection().getInfo()  # {'crs': 'EPSG:326xx', 'transform'..., 'nominalScale': 10}
        if isinstance(info, dict):
            crs = str(info.get("crs") or crs)
            ns = info.get("nominalScale")
            if isinstance(ns, (int, float)) and ns > 0:
                scale = int(round(float(ns)))
    except Exception:
        pass
    return crs, scale

def _download_image_to_path_patched(
    image: ee.Image,
    geometry: ee.Geometry,
    target: "Path",
    params: Optional[object] = None,   # caller may pass an object with .crs / .scale
    **_ignored,
) -> "_zones.ImageExportResult":
    """
    Export with the image's native CRS+scale. No reproject(), no dimensions.
    """
    # Prefer explicit params from caller; otherwise read from image itself.
    if params is not None and getattr(params, "crs", None) and getattr(params, "scale", None):
        crs = str(params.crs)
        scale = int(params.scale)
    else:
        crs, scale = _proj_info(image)

    # Ensure region is an ee.Geometry
    try:
        ee_region = geometry if isinstance(geometry, ee.Geometry) else ee.Geometry(_zones._geometry_region(geometry))
    except Exception:
        ee_region = geometry

    sanitized_name = _zones.sanitize_name(target.stem or "export")
    description = f"zones_{sanitized_name}"[:100]
    folder = os.getenv("GEE_DRIVE_FOLDER", "Sentinel2_Zones")

    log.info("EE export: image CRS=%s, scale=%sm, target=%s", crs, scale, sanitized_name)

    # Start optional Drive task (for metadata; we still stream a local file)
    task = None
    try:
        task = ee.batch.Export.image.toDrive(
            image=image,                # keep original projection/mask
            description=description,
            folder=folder,
            fileNamePrefix=sanitized_name,
            region=ee_region,           # ee.Geometry is fine here
            crs=crs,
            scale=scale,
            fileFormat="GeoTIFF",
            maxPixels=_zones.gee.MAX_PIXELS,
        )
        task.start()
    except Exception:
        _zones.logger.exception("Failed to start Drive export for %s", sanitized_name)
        task = None

    # Direct download (this is what the ZIP endpoint serves)
    dl_params = {
        "region": ee_region,
        "crs": crs,
        "scale": scale,
        "format": "GeoTIFF",
        "filePerBand": False,   # OK for direct download; do NOT send to Drive
        "noData": -32768,       # masked pixels stay masked (not 0)
    }
    url = image.getDownloadURL(dl_params)
    return _stream_download_to_path(url, target, is_zip_ok=True, task=task)

def _stream_download_to_path(url: str, target: "Path", *, is_zip_ok: bool, task) -> "_zones.ImageExportResult":
    """Stream HTTP body to disk; if ZIP, extract first GeoTIFF – no RAM spikes."""
    from zipfile import ZipFile
    from tempfile import NamedTemporaryFile
    with urlopen(url) as resp:
        content_type = getattr(resp, "headers", {}).get("Content-Type", "")
        if is_zip_ok and "zip" in str(content_type).lower():
            with NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
                shutil.copyfileobj(resp, tmp)
                tmp_path = Path(tmp.name)
            with ZipFile(tmp_path, "r") as zf:
                members = [m for m in zf.namelist() if m.lower().endswith((".tif", ".tiff"))]
                if not members:
                    raise ValueError("Zip archive did not contain a GeoTIFF file")
                with zf.open(members[0]) as zf_src, target.open("wb") as out_f:
                    shutil.copyfileobj(zf_src, out_f)
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
        else:
            with target.open("wb") as out_f:
                shutil.copyfileobj(resp, out_f)
    return _zones.ImageExportResult(path=target, task=task)

# Expose helper for any module that wants it
_zones._stream_download_to_path = _stream_download_to_path  # type: ignore[attr-defined]

# Apply the patch (idempotent)
if _ORIG_download_image_to_path and _ORIG_download_image_to_path is not _download_image_to_path_patched:
    _zones._download_image_to_path = _download_image_to_path_patched  # type: ignore[assignment]
    log.info("Patched zones._download_image_to_path – using image native CRS+scale")
else:
    log.warning("zones._download_image_to_path could not be patched (already patched or missing).")
