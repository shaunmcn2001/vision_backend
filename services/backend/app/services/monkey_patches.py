"""
Exporter patch v4 (dimensions-forced):
- No image.reproject() (keeps masks/values intact).
- Uses ee.Geometry directly for region.
- Forces CRS=EPSG:3857 and 10 m pixels.
- Additionally computes explicit image WIDTHxHEIGHT in pixels from AOI bounds in that CRS,
  and passes `dimensions="<w>x<h>"` to export. This prevents Earth Engine from collapsing
  to a single pixel due to projection/scale quirks.
- Sets noData=-32768 for direct download so masked pixels are not zeros.
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

def _aoi_pixel_dimensions(geometry: ee.Geometry, *, crs: str, scale: int) -> tuple[int, int]:
    """Compute width/height in pixels from AOI bounds in the given CRS."""
    proj = ee.Projection(crs)
    # Transform AOI to target CRS and get bounds polygon
    bounds = ee.Geometry(geometry).transform(proj, 1).bounds(1)
    coords = bounds.coordinates().getInfo()[0]  # outer ring
    xs = [pt[0] for pt in coords]
    ys = [pt[1] for pt in coords]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    width_m = max(0.0, float(maxx - minx))
    height_m = max(0.0, float(maxy - miny))
    # Ensure at least 1 pixel but typically many
    w = max(1, int(round(width_m / float(scale))))
    h = max(1, int(round(height_m / float(scale))))
    return w, h

def _download_image_to_path_patched(
    image: ee.Image,
    geometry: ee.Geometry,
    target: "Path",
    params: Optional[object] = None,
    **_ignored,
) -> "_zones.ImageExportResult":
    crs = getattr(params, "crs", DEFAULT_EXPORT_CRS)
    scale = int(getattr(params, "scale", DEFAULT_SCALE))

    # region as ee.Geometry
    ee_region = geometry if isinstance(geometry, ee.Geometry) else ee.Geometry(_zones._geometry_region(geometry))

    # Decide pixel grid size explicitly
    try:
        width_px, height_px = _aoi_pixel_dimensions(ee_region, crs=crs, scale=scale)
    except Exception:
        width_px, height_px = None, None

    sanitized_name = _zones.sanitize_name(target.stem or "export")
    description = f"zones_{sanitized_name}"[:100]
    folder = os.getenv("GEE_DRIVE_FOLDER", "Sentinel2_Zones")

    # Start Drive export (optional task id for metadata)
    task = None
    try:
        task_kwargs = dict(
            image=image,                  # keep original projection/mask
            description=description,
            folder=folder,
            fileNamePrefix=sanitized_name,
            region=ee_region,
            crs=crs,
            fileFormat="GeoTIFF",
            maxPixels=_zones.gee.MAX_PIXELS,
        )
        # Prefer explicit dimensions if we could compute them; else fall back to scale
        if width_px and height_px:
            task_kwargs["dimensions"] = f"{width_px}x{height_px}"
        else:
            task_kwargs["scale"] = scale
        task = ee.batch.Export.image.toDrive(**task_kwargs)
        task.start()
    except Exception:
        _zones.logger.exception("Failed to start Drive export for %s", sanitized_name)
        task = None

    # Direct download (stream to disk)
    dl_params = {
        "region": ee_region,
        "crs": crs,
        "format": "GeoTIFF",
        "filePerBand": False,
        "noData": -32768,
    }
    if width_px and height_px:
        dl_params["dimensions"] = f"{width_px}x{height_px}"
    else:
        dl_params["scale"] = scale

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

# Apply patch idempotently
if _ORIG_download_image_to_path and _ORIG_download_image_to_path is not _download_image_to_path_patched:
    _zones._download_image_to_path = _download_image_to_path_patched  # type: ignore[assignment]
    log.info("Patched zones._download_image_to_path (CRS=%s, ~10m, explicit dimensions)", DEFAULT_EXPORT_CRS)
else:
    log.warning("zones._download_image_to_path could not be patched (already patched or missing).")
