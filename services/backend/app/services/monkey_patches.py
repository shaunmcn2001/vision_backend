"""
Monkey patches for app.services.zones – fixes constant-value GeoTIFF exports and
accepts an optional `params` argument (with `.crs` and `.scale`), matching the
zones._download_image_to_path signature.
"""
from __future__ import annotations

import io
import logging
import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional

from urllib.request import urlopen
import ee

# Import the live zones module we want to patch
from app.services import zones as _zones

log = logging.getLogger(__name__)

# Re-export for static checkers / defaults
DEFAULT_SCALE: int = getattr(_zones, "DEFAULT_SCALE", 10)
DEFAULT_EXPORT_CRS: str = getattr(_zones, "DEFAULT_EXPORT_CRS", "EPSG:3857")

def _safe_reproject_for_export(image: ee.Image, *, is_categorical: bool, crs: str, scale: int) -> ee.Image:
    """Reproject to `crs` at `scale` meters.
    - nearest for classes; bilinear for continuous.
    """
    try:
        resampled = image.resample("nearest" if is_categorical else "bilinear")
        proj = ee.Projection(crs)
        return resampled.reproject(proj, None, scale)
    except Exception:  # defensive: keep exports working if EE errors
        return image

# Keep a reference to the original (in case we need to restore).
_ORIG_download_image_to_path = getattr(_zones, "_download_image_to_path", None)

def _download_image_to_path_patched(
    image: ee.Image,
    geometry: ee.Geometry,
    target: "Path",
    params: Optional[object] = None,
    **_ignored,
) -> "_zones.ImageExportResult":
    """Patched exporter:
    - accepts optional `params` (with `.crs` and `.scale`)
    - forces metric CRS/scale for Drive and direct download
    """
    # pull desired crs/scale from params if provided
    crs = getattr(params, "crs", DEFAULT_EXPORT_CRS)
    scale = int(getattr(params, "scale", DEFAULT_SCALE))

    # Heuristic: consider images named 'zone' or integer type as categorical
    try:
        band_names = image.bandNames()
        first_name = ee.String(band_names.get(0))
        name = first_name.getInfo() if hasattr(first_name, "getInfo") else "unknown"
        is_categorical = str(name).lower() in {"zone", "zones", "class", "classes"}
    except Exception:
        is_categorical = False

    img_for_export = _safe_reproject_for_export(image, is_categorical=is_categorical, crs=crs, scale=scale)

    sanitized_name = _zones.sanitize_name(target.stem or "export")
    description = f"zones_{sanitized_name}"[:100]
    folder = os.getenv("GEE_DRIVE_FOLDER", "Sentinel2_Zones")

    # Start a Drive export (optional – status JSON will include the task id)
    task = None
    try:
        task = ee.batch.Export.image.toDrive(
            image=img_for_export,
            description=description,
            folder=folder,
            fileNamePrefix=sanitized_name,
            region=_zones._geometry_region(geometry),
            scale=scale,
            crs=crs,
            fileFormat="GeoTIFF",
            maxPixels=_zones.gee.MAX_PIXELS,
        )
        task.start()
    except Exception:  # defensive
        _zones.logger.exception("Failed to start Drive export for %s", sanitized_name)
        task = None

    # Always stream a local artifact via getDownloadURL (this is what your ZIP download serves)
    dl_params = {
        "region": _zones._geometry_region(geometry),
        "scale": scale,
        "crs": crs,             # critical fix (avoid degrees)
        "format": "GeoTIFF",
        "filePerBand": False,   # OK for direct download; do NOT send to Drive
    }
    url = img_for_export.getDownloadURL(dl_params)
    return _stream_download_to_path(url, target, is_zip_ok=True, filename_hint=sanitized_name, task=task)

def _stream_download_to_path(url: str, target: "Path", *, is_zip_ok: bool, filename_hint: str, task) -> "_zones.ImageExportResult":
    with urlopen(url) as resp:
        content_type = getattr(resp, "headers", {}).get("Content-Type", "")
        if is_zip_ok and "zip" in str(content_type).lower():
            import tempfile
            from zipfile import ZipFile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
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

# Attach helper so zones can reuse if needed
_zones._stream_download_to_path = _stream_download_to_path  # type: ignore[attr-defined]

# Apply the patch (idempotent)
if _ORIG_download_image_to_path and _ORIG_download_image_to_path is not _download_image_to_path_patched:
    _zones._download_image_to_path = _download_image_to_path_patched  # type: ignore[assignment]
    log.info("Patched zones._download_image_to_path (CRS=%s, scale=%sm) – accepts `params`", DEFAULT_EXPORT_CRS, DEFAULT_SCALE)
else:
    log.warning("zones._download_image_to_path could not be patched (already patched or missing).")
