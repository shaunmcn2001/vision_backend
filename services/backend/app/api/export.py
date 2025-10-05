import io
import zipfile
from datetime import date, datetime, timedelta
from typing import Dict, Iterator, List, Mapping, Optional, Tuple
import urllib.error
import urllib.request

import ee
from ee.ee_exception import EEException
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from app import exports, index_visualization, indices as sentinel_indices
from app.services.image_stats import temporal_stats
from app.services.indices import UnsupportedIndexError, resolve_index
from app.services.tiles import init_ee
from app.utils.shapefile import shapefile_zip_to_geojson


DEFAULT_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
_CLOUD_COVER_THRESHOLD = 60


router = APIRouter()
sentinel2_router = APIRouter(prefix="/export/s2", tags=["sentinel-2"])


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


def _index_collection_for_range(
    geometry_geojson: dict,
    start_iso: str,
    end_iso: str,
    *,
    definition,
    parameters: Mapping[str, object],
    collection_name: str = DEFAULT_COLLECTION,
):
    geom = ee.Geometry(geometry_geojson)

    def _map_index_band(img):
        image = ee.Image(img)
        index_band = ee.Image(definition.compute(image, parameters))
        return image.addBands(index_band)

    collection = (
        ee.ImageCollection(collection_name)
        .filterBounds(geom)
        .filterDate(start_iso, end_iso)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", _CLOUD_COVER_THRESHOLD))
        .map(_map_index_band)
    )
    return geom, collection


def _index_image_for_range(
    geometry_geojson: dict,
    start_iso: str,
    end_iso: str,
    *,
    definition,
    parameters: Mapping[str, object],
    collection_name: str = DEFAULT_COLLECTION,
) -> Tuple[ee.ImageCollection, ee.Image]:
    geom, collection = _index_collection_for_range(
        geometry_geojson,
        start_iso,
        end_iso,
        definition=definition,
        parameters=parameters,
        collection_name=collection_name,
    )
    index_collection = collection.select(definition.band_name)
    stats = temporal_stats(
        index_collection,
        band_name=definition.band_name,
        rename_prefix=definition.band_name,
        mean_band_name=definition.band_name,
    )
    mean_image = stats["mean"]
    image = mean_image.clip(geom)
    if definition.valid_range is not None:
        low, high = definition.valid_range
        image = image.clamp(low, high)
    return collection, image


@router.post("/export")
async def export_geotiffs(
    start_date: str = Form(...),
    end_date: str = Form(...),
    file: UploadFile = File(..., description="Shapefile ZIP archive"),
    index: str = Form("ndvi"),
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

    geometry_ee = ee.Geometry(geometry)

    try:
        definition, resolved_parameters = resolve_index(index)
    except UnsupportedIndexError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

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
                collection, image = _index_image_for_range(
                    geometry,
                    start_iso,
                    end_iso,
                    definition=definition,
                    parameters=resolved_parameters,
                )
            except EEException as exc:
                raise HTTPException(
                    status_code=502,
                    detail=f"Failed to build {definition.band_name} image for {year}-{month:02d}: {exc}",
                ) from exc

            size = await run_in_threadpool(_collection_size, collection)
            if size == 0:
                raise HTTPException(
                    status_code=404,
                    detail=f"No {definition.code.upper()} imagery available for {year}-{month:02d}.",
                )

            prepared_image, is_visualized = index_visualization.prepare_image_for_export(
                image,
                definition.band_name,
                geometry_ee,
                definition.default_scale,
            )

            params: Dict[str, object] = {
                "scale": definition.default_scale,
                "region": geometry_ee,
                "filePerBand": False,
                "format": "GEO_TIFF",
            }
            format_options: Dict[str, object] = {"cloudOptimized": False}
            if not is_visualized:
                params["noDataValue"] = -9999
                format_options["noDataValue"] = -9999
            params["formatOptions"] = format_options

            try:
                url = prepared_image.getDownloadURL(params)
            except EEException as exc:
                raise HTTPException(
                    status_code=502,
                    detail=f"Failed to create download URL for {definition.code.upper()} {year}-{month:02d}: {exc}",
                ) from exc

            payload, content_type = await run_in_threadpool(_download_bytes, url)
            tif_bytes = _extract_tif(payload, content_type)

            output_name = f"{definition.code}_{year}_{month:02d}.tif"
            combined_zip.writestr(output_name, tif_bytes)
            months_written += 1

    if months_written == 0:
        raise HTTPException(status_code=404, detail="No months found within the requested date range.")

    output_buffer.seek(0)
    disposition = (
        f'attachment; filename="{definition.code}_{start.isoformat()}_{end.isoformat()}.zip"'
    )
    headers: Dict[str, str] = {
        "Content-Disposition": disposition,
        "X-Vegetation-Index": definition.code,
    }
    return StreamingResponse(output_buffer, media_type="application/zip", headers=headers)


def _derive_month_list(start: date, end: date) -> List[str]:
    months: List[str] = []
    for year, month, _, _ in _iter_month_ranges(start, end):
        label = f"{year:04d}-{month:02d}"
        if label not in months:
            months.append(label)
    return months


def _normalise_indices(values: List[str]) -> List[str]:
    if not values:
        raise HTTPException(status_code=400, detail="At least one index must be specified.")

    canonical_lookup = {
        name.lower(): name for name in sentinel_indices.SUPPORTED_INDICES
    }

    cleaned: List[str] = []
    for value in values:
        key = str(value).strip()
        canonical = canonical_lookup.get(key.lower())
        if canonical is None:
            raise HTTPException(status_code=400, detail=f"Unsupported index: {value}")
        if canonical not in cleaned:
            cleaned.append(canonical)
    return cleaned


@sentinel2_router.post("/indices/upload")
async def queue_sentinel2_exports(
    start_date: str = Form(...),
    end_date: str = Form(...),
    indices: List[str] = Form(...),
    file: UploadFile = File(..., description="Shapefile ZIP archive"),
    aoi_name: str = Form(""),
    export_target: str = Form("zip"),
    scale_m: int = Form(10),
    cloud_prob_max: int = Form(40),
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
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail=f"Failed to parse shapefile ZIP: {exc}") from exc

    months = _derive_month_list(start, end)
    if not months:
        raise HTTPException(status_code=400, detail="No months found within the requested date range.")

    index_list = _normalise_indices(indices)

    target = export_target.strip().lower() if isinstance(export_target, str) else "zip"
    if target not in {"zip", "gcs", "drive"}:
        raise HTTPException(status_code=400, detail="export_target must be one of: zip, gcs, drive.")

    if scale_m <= 0:
        raise HTTPException(status_code=400, detail="scale_m must be positive.")
    if cloud_prob_max < 0 or cloud_prob_max > 100:
        raise HTTPException(status_code=400, detail="cloud_prob_max must be between 0 and 100.")

    try:
        job = exports.create_job(
            aoi_geojson=geometry,
            months=months,
            index_names=index_list,
            export_target=target,
            aoi_name=aoi_name,
            scale_m=scale_m,
            cloud_prob_max=cloud_prob_max,
        )
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail=f"Failed to queue export: {exc}") from exc

    return {"job_id": job.job_id, "state": job.state}
