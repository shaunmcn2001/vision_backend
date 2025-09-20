import io
import logging
import zipfile
from datetime import date, datetime, timedelta
from typing import Iterator, Optional, Tuple
import urllib.error
import urllib.request

import ee
from ee.ee_exception import EEException
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from app.services.tiles import init_ee
from app.utils.shapefile import shapefile_zip_to_geojson


router = APIRouter()


def _parse_iso_date(label: str, value: str) -> date:
    try:
        return datetime.fromisoformat(value).date()
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"{label} must be an ISO 8601 date (YYYY-MM-DD)."
        ) from exc


def _month_start(d: date) -> date:
    return date(d.year, d.month, 1)


def _next_month(d: date) -> date:
    return (d.replace(day=28) + timedelta(days=4)).replace(day=1)


def _iter_month_ranges(start: date, end: date) -> Iterator[Tuple[int, int, date, date]]:
    current = _month_start(start)
    last = _month_start(end)
    end_exclusive = end + timedelta(days=1)

    while current <= last:
        following = _next_month(current)
        period_start = max(start, current)
        period_end = min(end_exclusive, following)
        if period_start < period_end:
            yield current.year, current.month, period_start, period_end
        current = following


def _collection_size(collection: ee.ImageCollection) -> int:
    return int(collection.size().getInfo())


def _download_bytes(url: str) -> Tuple[bytes, Optional[str]]:
    try:
        with urllib.request.urlopen(url) as resp:
            if resp.status != 200:
                raise HTTPException(status_code=502, detail=f"Earth Engine download failed with status {resp.status}.")
            return resp.read(), resp.headers.get("Content-Type")
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=502, detail=f"Failed to download GeoTIFF: {exc.reason}") from exc


def _has_tiff_magic(payload: bytes) -> bool:
    if len(payload) < 4:
        return False
    header = payload[:4]
    return header in (b"II*\x00", b"MM\x00*")


def _looks_like_tiff(content_type: Optional[str], payload: bytes) -> bool:
    if content_type and "tif" in content_type.lower():
        return True
    return _has_tiff_magic(payload)


def _extract_tif(payload: bytes, content_type: Optional[str]) -> bytes:
    if _looks_like_tiff(content_type, payload):
        return payload

    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            tif_names = [name for name in archive.namelist() if name.lower().endswith(".tif")]
            if not tif_names:
                raise HTTPException(status_code=502, detail="Earth Engine download did not contain a GeoTIFF.")
            return archive.read(tif_names[0])
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=502, detail="Earth Engine response was not a valid ZIP archive.") from exc


def _ndvi_image_for_range(geometry_geojson: dict, start_iso: str, end_iso: str) -> Tuple[ee.ImageCollection, ee.Image]:
    geom = ee.Geometry(geometry_geojson)
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geom)
        .filterDate(start_iso, end_iso)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
        .map(lambda img: img.addBands(img.normalizedDifference(["B8", "B4"]).rename("NDVI")))
    )
    image = collection.select("NDVI").mean().clip(geom)
    image = image.clamp(-1, 1)
    return collection, image


@router.post("/export")
async def export_geotiffs(
    start_date: str = Form(...),
    end_date: str = Form(...),
    file: UploadFile = File(..., description="Shapefile ZIP archive"),
    source_epsg: Optional[str] = Form(
        None,
        description="EPSG code of the uploaded shapefile when no .prj is included.",
    ),
):
    filename = (file.filename or "").lower()
    if not filename.endswith(".zip"):
        raise HTTPException(status_code=415, detail="Upload must be a shapefile ZIP archive (.zip).")

    start = _parse_iso_date("start_date", start_date)
    end = _parse_iso_date("end_date", end_date)
    if start > end:
        raise HTTPException(status_code=400, detail="start_date must be on or before end_date.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded shapefile ZIP is empty.")

    epsg_code = source_epsg.strip() if source_epsg else None

    try:
        geometry, defaulted_crs = shapefile_zip_to_geojson(content, source_epsg=epsg_code)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse shapefile ZIP: {exc}") from exc

    if defaulted_crs:
        # Export endpoint returns binary data, so we log instead of modifying the response.
        logging.getLogger(__name__).warning(
            "Shapefile missing CRS; defaulted to EPSG:4326 (WGS84) for export request."
        )

    try:
        init_ee()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to initialize Earth Engine: {exc}") from exc

    output_buffer = io.BytesIO()
    months_written = 0

    with zipfile.ZipFile(output_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as combined_zip:
        for year, month, period_start, period_end in _iter_month_ranges(start, end):
            start_iso = period_start.isoformat()
            end_iso = period_end.isoformat()

            try:
                collection, image = _ndvi_image_for_range(geometry, start_iso, end_iso)
            except EEException as exc:
                raise HTTPException(status_code=502, detail=f"Failed to build NDVI image for {year}-{month:02d}: {exc}") from exc

            size = await run_in_threadpool(_collection_size, collection)
            if size == 0:
                raise HTTPException(status_code=404, detail=f"No imagery available for {year}-{month:02d}.")

            try:
                url = image.getDownloadURL(
                    {
                        "scale": 10,
                        "region": geometry,
                        "filePerBand": False,
                        "format": "GEO_TIFF",
                    }
                )
            except EEException as exc:
                raise HTTPException(status_code=502, detail=f"Failed to create download URL for {year}-{month:02d}: {exc}") from exc

            payload, content_type = await run_in_threadpool(_download_bytes, url)
            tif_bytes = _extract_tif(payload, content_type)

            output_name = f"ndvi_{year}_{month:02d}.tif"
            combined_zip.writestr(output_name, tif_bytes)
            months_written += 1

    if months_written == 0:
        raise HTTPException(status_code=404, detail="No months found within the requested date range.")

    output_buffer.seek(0)
    disposition = f'attachment; filename="ndvi_{start.isoformat()}_{end.isoformat()}.zip"'
    headers = {"Content-Disposition": disposition}
    return StreamingResponse(output_buffer, media_type="application/zip", headers=headers)
