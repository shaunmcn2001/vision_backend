"""
Exporter patch v3:
- No image.reproject() (keeps masks intact).
- Uses the *ee.Geometry* directly for region (safer than raw coordinates).
- Forces CRS=EPSG:3857, scale=10m for both Drive and direct download.
- Sets noData on direct download so masked pixels aren't written as zeros.
- Accepts optional `params` with `.crs` and `.scale`.
"""
from __future__ import annotations
import logging
import os
import shutil
from pathlib import Path
from typing import Optional
from urllib.request import urlopen

import ee
from app.services import zones as _zones

log = logging.getLogger(__name__)

DEFAULT_SCALE: int = getattr(_zones, "DEFAULT_SCALE", 10)
DEFAULT_EXPORT_CRS: str = getattr(_zones, "DEFAULT_EXPORT_CRS", "EPSG:3857")

_ORIG_download_image_to_path = getattr(_zones, "_download_image_to_path", None)

def _download_image_to_path_patched(
    image: ee.Image,
    geometry: ee.Geometry,
    target: "Path",
    params: Optional[object] = None,
    **_ignored,
) -> "_zones.ImageExportResult":
    crs = getattr(params, "crs", DEFAULT_EXPORT_CRS)
    scale = int(getattr(params, "scale", DEFAULT_SCALE))

    # Ensure region is an ee.Geometry
    try:
        ee_region = geometry if isinstance(geometry, ee.Geometry) else ee.Geometry(_zones._geometry_region(geometry))
    except Exception:
        ee_region = geometry

    sanitized_name = _zones.sanitize_name(target.stem or "export")
    description = f"zones_{sanitized_name}"[:100]
    folder = os.getenv("GEE_DRIVE_FOLDER", "Sentinel2_Zones")

    # Start Drive export (optional; task id shows in metadata)
    task = None
    try:
        task = ee.batch.Export.image.toDrive(
            image=image,                  # keep original projection/mask
            description=description,
            folder=folder,
            fileNamePrefix=sanitized_name,
            region=ee_region,
            scale=scale,
            crs=crs,
            fileFormat="GeoTIFF",
            maxPixels=_zones.gee.MAX_PIXELS,
        )
        task.start()
    except Exception:
        _zones.logger.exception("Failed to start Drive export for %s", sanitized_name)
        task = None

    # Stream local artifact via getDownloadURL (explicit CRS/scale + noData)
    dl_params = {
        "region": ee_region,
        "scale": scale,
        "crs": crs,
        "format": "GeoTIFF",
        "filePerBand": False,  # OK for direct download
        "noData": -32768,      # make masked pixels obvious (not zero)
    }
    url = image.getDownloadURL(dl_params)
    return _stream_download_to_path(url, target, is_zip_ok=True, task=task)

def _stream_download_to_path(url: str, target: "Path", *, is_zip_ok: bool, task) -> "_zones.ImageExportResult":
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
            try: tmp_path.unlink(missing_ok=True)
            except Exception: pass
        else:
            with target.open("wb") as out_f:
                shutil.copyfileobj(resp, out_f)
    return _zones.ImageExportResult(path=target, task=task)

# Expose helper
_zones._stream_download_to_path = _stream_download_to_path  # type: ignore[attr-defined]

# Apply patch idempotently
if _ORIG_download_image_to_path and _ORIG_download_image_to_path is not _download_image_to_path_patched:
    _zones._download_image_to_path = _download_image_to_path_patched  # type: ignore[assignment]
    log.info("Patched zones._download_image_to_path (CRS=%s, scale=%sm, no reproject, noData=-32768)", DEFAULT_EXPORT_CRS, DEFAULT_SCALE)
else:
    log.warning("zones._download_image_to_path could not be patched (already patched or missing).")
