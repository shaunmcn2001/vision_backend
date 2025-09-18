import io
import zipfile
from datetime import date, datetime, timedelta
from typing import Iterator, Tuple
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


def _download_bytes(url: str) -> bytes:
    try:
        with urllib.request.urlopen(url) as resp:
            if resp.status != 200:
                raise HTTPException(status_code=502, detail=f"Earth Engine download failed with status {resp.status}.")
            return resp.read()
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=502, detail=f"Failed to download GeoTIFF: {exc.reason}") from exc


def _extract_first_tif(zip_bytes: bytes) -> bytes:
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
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
    image = image.updateMask(image.gte(0).And(image.lte(1)))
    return collection, image


@router.post("/export")
async def export_geotiffs(
    start_date: str = Form(...),
    end_date: str = Form(...),
    file: UploadFile = File(..., description="Shapefile ZIP archive"),
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

    try:
        geometry = shapefile_zip_to_geojson(content)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse shapefile ZIP: {exc}") from exc

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

            zip_bytes = await run_in_threadpool(_download_bytes, url)
            tif_bytes = _extract_first_tif(zip_bytes)

            output_name = f"ndvi_{year}_{month:02d}.tif"
            combined_zip.writestr(output_name, tif_bytes)
            months_written += 1

    if months_written == 0:
        raise HTTPException(status_code=404, detail="No months found within the requested date range.")

    output_buffer.seek(0)
    disposition = f'attachment; filename="ndvi_{start.isoformat()}_{end.isoformat()}.zip"'
    headers = {"Content-Disposition": disposition}
    return StreamingResponse(output_buffer, media_type="application/zip", headers=headers)
