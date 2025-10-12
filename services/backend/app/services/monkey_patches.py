"""
Monkey patches for app.services.zones to fix constant-value GeoTIFF exports.

What this does
--------------
1) Forces a metric CRS (EPSG:3857) and 10 m scale when using getDownloadURL
   to avoid Earth Engine defaulting to degrees (which collapses the raster
   into a few huge pixels and looks "constant").
2) Also sets the CRS+scale on the Drive export task.
3) Uses nearest-neighbour resampling for classified rasters; bilinear for continuous.

How to enable (one-time)
------------------------
Add this import near the top of `services/backend/app/main.py` (after other imports):

    import app.services.monkey_patches  # noqa: F401

No other changes are needed.
"""
from __future__ import annotations

import logging
from typing import Optional

import ee

# Import the live zones module we want to patch
from app.services import zones as _zones

log = logging.getLogger(__name__)

# Re-export for static checkers
DEFAULT_SCALE: int = getattr(_zones, "DEFAULT_SCALE", 10)
DEFAULT_EXPORT_CRS: str = getattr(_zones, "DEFAULT_EXPORT_CRS", "EPSG:3857")

def _safe_reproject_for_export(image: ee.Image, *, is_categorical: bool) -> ee.Image:
    """Return image reprojected to a metric CRS at DEFAULT_SCALE.
    - nearest for classes; bilinear for continuous.
    - never raises; falls back to the original image on EE errors.
    """
    try:
        resampled = image.resample("nearest" if is_categorical else "bilinear")
        proj = ee.Projection(DEFAULT_EXPORT_CRS)
        # NOTE: third arg is nominalScale. We explicitly pass DEFAULT_SCALE (meters).
        return resampled.reproject(proj, None, DEFAULT_SCALE)
    except Exception:  # pragma: no cover – defensive: keep exports working
        return image

# Keep a reference to the original (in case we need to restore).
_ORIG_download_image_to_path = getattr(_zones, "_download_image_to_path", None)

def _download_image_to_path_patched(
    image: ee.Image, geometry: ee.Geometry, target: "Path"
) -> "_zones.ImageExportResult":
    """Patched exporter:
    - reprojects image to EPSG:3857 @ 10 m
    - sets CRS+scale for both Drive export and direct download
    """
    # Heuristic: consider images named 'zone' or integer type as categorical
    try:
        band_names = image.bandNames()
        first_name = ee.String(band_names.get(0))
        name = first_name.getInfo() if hasattr(first_name, "getInfo") else "unknown"
        is_categorical = str(name).lower() in {"zone", "zones", "class", "classes"}
    except Exception:
        is_categorical = False

    img_for_export = _safe_reproject_for_export(image, is_categorical=is_categorical)

    sanitized_name = _zones.sanitize_name(target.stem or "export")
    description = f"zones_{sanitized_name}"[:100]
    folder = _zones.os.getenv("GEE_DRIVE_FOLDER", "Sentinel2_Zones")

    # Try to start a Drive export (optional – status JSON will include the task id)
    task = None
    try:
        task = ee.batch.Export.image.toDrive(
            image=img_for_export,
            description=description,
            folder=folder,
            fileNamePrefix=sanitized_name,
            region=_zones._geometry_region(geometry),
            scale=DEFAULT_SCALE,
            crs=DEFAULT_EXPORT_CRS,
            fileFormat="GeoTIFF",
            maxPixels=_zones.gee.MAX_PIXELS,
        )
        task.start()
    except Exception:  # pragma: no cover
        _zones.logger.exception("Failed to start Drive export for %s", sanitized_name)
        task = None

    # Always stream a local artifact via getDownloadURL (this is what your ZIP download serves)
    params = {
        "region": _zones._geometry_region(geometry),
        "scale": DEFAULT_SCALE,
        "crs": DEFAULT_EXPORT_CRS,        # ← critical fix (avoid degrees)
        "format": "GeoTIFF",
        # DO NOT pass filePerBand to Drive (EE rejects it), but it is fine here.
        "filePerBand": False,
    }
    url = img_for_export.getDownloadURL(params)
    return _zones._stream_download_to_path(url, target, is_zip_ok=True, filename_hint=sanitized_name, task=task)

# We need a tiny helper to stream the HTTP response to disk without loading in RAM.
# Use the same semantics the zones module already relied on; if you already have a helper, we reuse it.
import io
import shutil
import zipfile
from urllib.request import urlopen

def _stream_download_to_path(url: str, target: "Path", *, is_zip_ok: bool, filename_hint: str, task) -> "_zones.ImageExportResult":
    with urlopen(url) as resp:
        content_type = getattr(resp, "headers", {}).get("Content-Type", "")
        if is_zip_ok and "zip" in str(content_type).lower():
            # Write ZIP to a temp file on disk and extract first GeoTIFF
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

    # Return the same shape as the original helper
    return _zones.ImageExportResult(path=target, task=task)

# Attach helper so zones can reuse if needed
_zones._stream_download_to_path = _stream_download_to_path  # type: ignore[attr-defined]

# Apply the patch (idempotent)
if _ORIG_download_image_to_path and _ORIG_download_image_to_path is not _download_image_to_path_patched:
    _zones._download_image_to_path = _download_image_to_path_patched  # type: ignore[assignment]
    log.info("Patched zones._download_image_to_path with metric-CRS exporter (CRS=%s, scale=%sm)", DEFAULT_EXPORT_CRS, DEFAULT_SCALE)
else:
    log.warning("zones._download_image_to_path could not be patched (already patched or missing).")
