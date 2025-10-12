from __future__ import annotations

import calendar
import csv
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union
from urllib.request import urlopen
from zipfile import ZipFile

import ee
import numpy as np
import rasterio
from rasterio.windows import Window

from app import gee
from app.exports import sanitize_name
from app.services.ndvi_shared import (
    compute_ndvi_loose,
    mean_from_collection_sum_count,
    reproject_native_10m,
)
from app.utils.sanitization import sanitize_for_json

logger = logging.getLogger(__name__)

DEFAULT_CLOUD_PROB_MAX = 100
DEFAULT_N_CLASSES = 5
DEFAULT_CV_THRESHOLD = 0.8
DEFAULT_MIN_MAPPING_UNIT_HA = 1
DEFAULT_SMOOTH_RADIUS_M = 0
DEFAULT_OPEN_RADIUS_M = 0
DEFAULT_CLOSE_RADIUS_M = 0
DEFAULT_SIMPLIFY_TOL_M = 5
DEFAULT_SIMPLIFY_BUFFER_M = 3
DEFAULT_SMOOTH_RADIUS_PX = 1
DEFAULT_METHOD = "ndvi_percentiles"
DEFAULT_SAMPLE_SIZE = 4000
DEFAULT_SCALE = int(os.getenv("ZONES_SCALE_M", "10"))
DEFAULT_EXPORT_CRS = "EPSG:3857"
DEFAULT_CRS = DEFAULT_EXPORT_CRS

ZONE_PALETTE: Tuple[str, ...] = (
    "#112f1d",
    "#1b4d2a",
    "#2c6a39",
    "#3f8749",
    "#58a35d",
    "#80bf7d",
    "#b6dcb1",
)

NDVI_MASK_EMPTY_ERROR = (
    "No valid NDVI pixels across the selected months. Try a wider date range or relax cloud masking."
)
NDVI_VARIATION_TOO_LOW_ERROR = (
    "NDVI variation too low to produce meaningful production zones. "
    "Try using a different time period or crop stage with greater contrast."
)

S2_COLLECTION_EMPTY_ERROR = (
    "No Sentinel-2 imagery found for the selected area and dates. "
    "Ensure your AOI and date range are within valid coverage."
)


@dataclass(frozen=True)
class ZoneArtifacts:
    raster_path: str
    mean_ndvi_path: str
    vector_path: str
    vector_components: dict[str, str]
    zonal_stats_path: str | None = None
    working_dir: str | None = None


@dataclass(frozen=True)
class ImageExportResult:
    path: Path
    task: ee.batch.Task | None = None


def _allow_init_failure() -> bool:
    flag = os.getenv("GEE_ALLOW_INIT_FAILURE", "")
    return flag.strip().lower() in {"1", "true", "yes"}


def _to_ee_geometry(geojson: dict) -> ee.Geometry:
    try:
        geo_type = (geojson or {}).get("type", "")
        if geo_type == "Feature":
            return ee.Feature(geojson).geometry()
        if geo_type == "FeatureCollection":
            return ee.FeatureCollection(geojson).geometry()
        return ee.Geometry(geojson)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Invalid AOI GeoJSON: {exc}")


def _ensure_working_directory(path: os.PathLike[str] | str | None) -> Path:
    if path:
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        return directory
    return Path(tempfile.mkdtemp(prefix="zones_"))


def _as_lists(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_as_lists(item) for item in value]
    if isinstance(value, list):
        return [_as_lists(item) for item in value]
    return value


def _geometry_region(geometry: ee.Geometry) -> List[List[List[float]]]:
    info = geometry.getInfo()
    if not info:
        raise ValueError("Unable to resolve geometry information for download")
    geom_type = info.get("type")
    if geom_type == "Polygon":
        coords = info.get("coordinates")
        if coords:
            return _as_lists(coords)
        raise ValueError("Geometry information missing coordinates")
    if geom_type == "MultiPolygon":
        dissolved = geometry.dissolve()
        dissolved_info = dissolved.getInfo() or info
        if dissolved_info.get("type") == "Polygon":
            return _as_lists(dissolved_info.get("coordinates", []))
    coords = info.get("coordinates")
    if coords is not None:
        return _as_lists(coords)
    raise ValueError("Geometry information missing coordinates")


def _write_zip_geotiff_from_file(zip_path: Path, target: Path) -> None:
    with ZipFile(zip_path, "r") as archive:
        members = [m for m in archive.namelist() if m.lower().endswith((".tif", ".tiff"))]
        if not members:
            raise ValueError("Zip archive did not contain a GeoTIFF file")
        first = members[0]
        with archive.open(first) as source, target.open("wb") as output:
            shutil.copyfileobj(source, output)


@dataclass
class _DownloadParams:
    crs: str | None = DEFAULT_EXPORT_CRS
    scale: int = DEFAULT_SCALE


def _download_image_to_path(
    image: ee.Image,
    geometry: ee.Geometry,
    target: Path,
    params: _DownloadParams | None = None,
) -> ImageExportResult:
    params = params or _DownloadParams()
    to_float = getattr(image, "toFloat", None)
    if callable(to_float):
        image = to_float()

    try:
        region_candidate = geometry.geometry() if hasattr(geometry, "geometry") else geometry
    except Exception:
        region_candidate = geometry
    if isinstance(region_candidate, ee.Geometry):
        ee_region = region_candidate
    else:
        region_coords = _geometry_region(geometry)
        ee_region = ee.Geometry.Polygon(region_coords)

    sanitized_name = sanitize_name(target.stem or "export")
    description = f"zones_{sanitized_name}"[:100]
    folder = os.getenv("GEE_DRIVE_FOLDER", "Sentinel2_Zones")

    task: ee.batch.Task | None = None
    try:
        export_kwargs: Dict[str, object] = {
            "image": image,
            "description": description,
            "folder": folder,
            "fileNamePrefix": sanitized_name,
            "region": ee_region,
            "scale": params.scale,
            "fileFormat": "GeoTIFF",
            "maxPixels": gee.MAX_PIXELS,
        }
        if params.crs:
            export_kwargs["crs"] = params.crs
        task = ee.batch.Export.image.toDrive(**export_kwargs)
        task.start()
    except Exception:  # pragma: no cover - defensive guard for Drive failures
        logger.exception("Failed to start Drive export for %s", sanitized_name)
        task = None

    download_params: Dict[str, object] = {
        "scale": params.scale,
        "region": ee_region,
        "filePerBand": False,
        "format": "GeoTIFF",
    }
    if params.crs:
        download_params["crs"] = params.crs
    download_params["noData"] = -32768
    url = image.getDownloadURL(download_params)

    with urlopen(url) as response:
        headers = getattr(response, "headers", None)
        content_type = headers.get("Content-Type", "") if headers and hasattr(headers, "get") else ""
        if "zip" in content_type.lower():
            with NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
                shutil.copyfileobj(response, tmp_zip)
                tmp_zip_path = Path(tmp_zip.name)
            _write_zip_geotiff_from_file(tmp_zip_path, target)
            try:
                tmp_zip_path.unlink(missing_ok=True)
            except Exception:
                pass
        else:
            with target.open("wb") as output:
                shutil.copyfileobj(response, output)
    return ImageExportResult(path=target, task=task)


def _download_vector_to_path(
    vectors: ee.FeatureCollection,
    target: Path,
    *,
    file_format: str,
) -> None:
    fmt = file_format.lower()
    params = {
        "format": fmt,
        "filename": target.stem,
        "selectors": ["zone"],
        "crs": "EPSG:4326",
    }
    url = vectors.getDownloadURL(params)
    if fmt == "geojson" and target.suffix.lower() != ".geojson":
        target = target.with_suffix(".geojson")
    if fmt == "kml" and target.suffix.lower() != ".kml":
        target = target.with_suffix(".kml")
    target.parent.mkdir(parents=True, exist_ok=True)

    with urlopen(url) as response:
        headers = getattr(response, "headers", None)
        content_type = headers.get("Content-Type", "") if headers and hasattr(headers, "get") else ""
        if "zip" in content_type.lower():
            with NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
                shutil.copyfileobj(response, tmp_zip)
                tmp_zip_path = Path(tmp_zip.name)
            try:
                with ZipFile(tmp_zip_path, "r") as archive:
                    members = archive.namelist()
                    if not members:
                        raise ValueError("Vector download was empty")
                    first = members[0]
                    with archive.open(first) as member:
                        target.write_bytes(member.read())
            finally:
                try:
                    tmp_zip_path.unlink(missing_ok=True)
                except Exception:
                    pass
        else:
            target.write_bytes(response.read())


def _ordered_months(months: Sequence[str]) -> List[str]:
    unique: Dict[str, datetime] = {}
    for raw in months:
        month = str(raw).strip()
        if not month:
            continue
        try:
            parsed = datetime.strptime(month, "%Y-%m")
        except ValueError as exc:
            raise ValueError(f"Invalid month format: {raw}") from exc
        if month not in unique:
            unique[month] = parsed
    return [key for key, _ in sorted(unique.items(), key=lambda item: item[1])]


def _month_range_dates(months: Sequence[str]) -> tuple[date, date]:
    ordered = _ordered_months(months)
    if not ordered:
        raise ValueError("At least one month must be supplied")
    start_dt = datetime.strptime(ordered[0], "%Y-%m")
    end_dt = datetime.strptime(ordered[-1], "%Y-%m")
    start_day = date(start_dt.year, start_dt.month, 1)
    end_last_day = calendar.monthrange(end_dt.year, end_dt.month)[1]
    end_day = date(end_dt.year, end_dt.month, end_last_day)
    return start_day, end_day


def _months_from_dates(start_date: date, end_date: date) -> List[str]:
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")
    months: List[str] = []
    cursor = date(start_date.year, start_date.month, 1)
    end_cursor = date(end_date.year, end_date.month, 1)
    while cursor <= end_cursor:
        months.append(cursor.strftime("%Y-%m"))
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)
    return months


def _export_prefix(aoi_name: str, months: Sequence[str]) -> str:
    ordered = _ordered_months(months)
    start, end = ordered[0], ordered[-1]
    safe_name = sanitize_name(aoi_name or "aoi")
    return f"zones/PROD_{start.replace('-', '')}_{end.replace('-', '')}_{safe_name}_zones"


def export_prefix(aoi_name: str, months: Sequence[str]) -> str:
    return _export_prefix(aoi_name, months)


def _resolve_geometry(aoi: Union[dict, ee.Geometry]) -> ee.Geometry:
    try:
        if isinstance(aoi, ee.Geometry):
            return aoi
    except TypeError:  # pragma: no cover - ee.Geometry on old ee module can raise TypeError
        pass
    return gee.geometry_from_geojson(aoi)
    
def _build_mean_ndvi_for_zones(
    geom: ee.Geometry,
    start_date: date | str,
    end_date: date | str,
    *,
    months: Sequence[str] | None = None,
    cloud_prob_max: int = 60,
) -> ee.Image:
    """
    Build mean NDVI for the selected period.
    Raises friendly errors if no imagery, no valid pixels, or too little variation.
    """

    # Normalize dates -> ISO strings
    def _iso(d):
        if isinstance(d, date):
            return d.isoformat()
        return str(d)

    # 1) Monthly Sentinel-2 collection (your helper handles months/clouds)
    monthly = gee.monthly_sentinel2_collection(
        aoi=geom, start=_iso(start_date), end=_iso(end_date),
        months=months, cloud_prob_max=cloud_prob_max
    )

    # Collection size (server → client)
    try:
        ic_size = int(ee.Number(monthly.size()).getInfo() or 0)
    except Exception:
        ic_size = 0
    if ic_size == 0:
        raise ValueError(S2_COLLECTION_EMPTY_ERROR)

    # 2) NDVI per image (safe band presence check)
    def _ndvi(img: ee.Image) -> ee.Image:
        bnames = ee.List(img.bandNames())
        has_b4 = bnames.contains("B4")
        has_b8 = bnames.contains("B8")
        both = ee.Algorithms.And(has_b4, has_b8)

        ndvi_img = ee.Image(
            ee.Algorithms.If(
                both,
                img.normalizedDifference(["B8", "B4"]).rename("NDVI"),
                # keep masked; don’t fabricate values
                ee.Image.constant(float("nan")).rename("NDVI")
            )
        )
        return ndvi_img.updateMask(img.mask())

    monthly_ndvi = monthly.map(_ndvi)

    # 3) Mean NDVI
    ndvi_mean = monthly_ndvi.mean().rename("NDVI_mean").toFloat().clip(geom)

    # 4) Valid-pixel check (sum of mask > 0)
    try:
        mask_sum_dict = ndvi_mean.mask().reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geom,
            scale=40,
            bestEffort=True,
            maxPixels=1e9,
        )
        mask_sum = ee.Number(ee.Dictionary(mask_sum_dict).values().get(0)).getInfo()
    except Exception:
        mask_sum = None

    if mask_sum is None or mask_sum <= 0:
        raise ValueError(NDVI_MASK_EMPTY_ERROR)

    # 5) Variation check (avoid single-zone outcomes)
    try:
        std_dict = ndvi_mean.reduceRegion(
            reducer=ee.Reducer.stdDev(),
            geometry=geom,
            scale=40,
            bestEffort=True,
            maxPixels=1e9,
        )
        std_val = ee.Number(ee.Dictionary(std_dict).values().get(0)).getInfo()
        std_val = float(std_val if std_val is not None else 0.0)
    except Exception:
        std_val = 0.0

    if std_val < 0.01:
        raise ValueError(NDVI_VARIATION_TOO_LOW_ERROR)

    # 6) Reproject to native 10 m if possible
    first_img = ee.Image(monthly.first())
    ndvi_mean_native = ee.Image(
        ee.Algorithms.If(
            first_img,
            reproject_native_10m(ndvi_mean, first_img, ref_band="B8", scale=10),
            ndvi_mean,
        )
    )

    return ndvi_mean_native.rename("NDVI_mean").toFloat().clip(geom)

def _classify_smooth_and_polygonize(
    ndvi_mean_native: ee.Image,
    geom: ee.Geometry,
    *,
    n_zones=5,
    mmu_ha=1.0,
    smooth_radius_px=1,
):
    """
    Classifies NDVI into production zones, ensuring at least two valid classes.
    Handles flat NDVI distributions and avoids Image.constant(None) crashes.
    """

    # --- 1. Guard NDVI ---
    ndvi_safe = ee.Image(
        ee.Algorithms.If(
            ndvi_mean_native,
            ndvi_mean_native.unmask(0),
            ee.Image.constant(0)
        )
    ).select([0]).rename("NDVI_mean")

    # --- 2. Winsorize NDVI (clip extremes) ---
    q = ndvi_safe.reduceRegion(
        ee.Reducer.percentile([2, 98]),
        geometry=geom,
        scale=10,
        bestEffort=True,
        maxPixels=1e9,
    )
    p2 = ee.Number(ee.Algorithms.If(q.get("NDVI_mean_p2"), q.get("NDVI_mean_p2"), 0))
    p98 = ee.Number(ee.Algorithms.If(q.get("NDVI_mean_p98"), q.get("NDVI_mean_p98"), 1))
    ndvi_w = ndvi_safe.max(p2).min(p98)

    # --- 3. Quantiles ---
    cuts = ee.List.sequence(100.0 / n_zones, 100.0 - 100.0 / n_zones, 100.0 / n_zones)
    quants = ndvi_w.reduceRegion(
        ee.Reducer.percentile(cuts),
        geometry=geom,
        scale=10,
        bestEffort=True,
        maxPixels=1e9,
    )

    def safe_quantile(p):
        key = ee.String("NDVI_mean_p").cat(ee.Number(p).toInt().format())
        val = quants.get(key)
        return ee.Number(
            ee.Algorithms.If(
                ee.Algorithms.IsEqual(val, None),
                0.0,
                ee.Number(ee.Algorithms.If(val, val, 0.0))
            )
        )

    thresholds = ee.List(cuts.map(safe_quantile))

    # --- 4. Validate thresholds list ---
    thresholds_sum = ee.Number(thresholds.reduce(ee.Reducer.sum()))
    thresholds = ee.Algorithms.If(
        thresholds_sum.eq(0),
        ee.List.sequence(0.2, 0.8, 0.2),
        thresholds
    )
    thresholds = ee.List(thresholds)

    # --- 4b. Enforce at least two distinct thresholds ---
    def _ensure_two(thr_list):
        thr_list = ee.List(thr_list)
        min_thr = ee.Number(thr_list.reduce(ee.Reducer.min()))
        max_thr = ee.Number(thr_list.reduce(ee.Reducer.max()))
        range_ok = max_thr.subtract(min_thr).gt(0.001)
        return ee.List(
            ee.Algorithms.If(
                range_ok,
                thr_list,
                ee.List.sequence(0.3, 0.7, 0.4)  # fallback 2-zone spread
            )
        )

    thresholds = _ensure_two(thresholds)

    # --- 5. Classification ---
    # --- 5. Classification ---
    def classify_by_thresholds(img, thr_list):
        # Sanitize thresholds: replace nulls with +inf so img.gte(thr) -> False
        def _sanit(t):
            return ee.Number(ee.Algorithms.If(ee.Algorithms.IsEqual(t, None), 1e9, t))
        thr_list = ee.List(thr_list).map(_sanit)
    
        def _step(acc, t):
            tnum = ee.Number(t)
            return ee.Image(acc).add(img.gte(tnum))  # zones start at 1
    
        # iterate always returns an Image (the accumulator), never None
        zones_img = ee.Image(thr_list.iterate(_step, ee.Image.constant(1)))
        return zones_img.rename("zone").toInt8().updateMask(img.mask())
    
    cls_raw = classify_by_thresholds(ndvi_w, thresholds)
    
    # --- 6. Ensure at least 2 unique zone values (fallback: median split) ---
    hist_dict = cls_raw.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=geom,
        scale=40,
        bestEffort=True,
        maxPixels=1e9,
    )
    
    zone_hist = ee.Dictionary(ee.Algorithms.If(
        ee.Algorithms.IsEqual(hist_dict, None),
        ee.Dictionary({}),
        ee.Dictionary(hist_dict).get("zone")
    ))
    zone_hist = ee.Dictionary(ee.Algorithms.If(
        ee.Algorithms.IsEqual(zone_hist, None),
        ee.Dictionary({}),
        zone_hist
    ))
    zone_count = ee.Number(zone_hist.size())
    
    # Safe median (p50) with fallback to mean if needed
    q50 = ndvi_w.reduceRegion(
        ee.Reducer.percentile([50]),
        geometry=geom,
        scale=40,
        bestEffort=True,
        maxPixels=1e9,
    ).get("NDVI_mean_p50")
    
    mean_fallback = ndvi_w.reduceRegion(
        ee.Reducer.mean(),
        geometry=geom,
        scale=40,
        bestEffort=True,
        maxPixels=1e9,
    ).get("NDVI_mean")
    
    median_val = ee.Number(ee.Algorithms.If(
        ee.Algorithms.IsEqual(q50, None),
        ee.Algorithms.If(ee.Algorithms.IsEqual(mean_fallback, None), 0.0, mean_fallback),
        q50
    ))
    
    # If only one class, do a median split into 2 classes (1,2)
    cls_raw = ee.Image(ee.Algorithms.If(
        zone_count.lte(1),
        ndvi_w.gt(median_val).add(1).toInt8().updateMask(ndvi_w.mask()).rename("zone"),
        cls_raw
    )).toInt8().rename("zone")
    
    # --- 7. Optional smoothing ---
    cls_smooth = ee.Image(ee.Algorithms.If(
        ee.Number(smooth_radius_px).gt(0),
        cls_raw.focalMode(radius=smooth_radius_px, units="pixels"),
        cls_raw
    )).toInt8()
    
    # --- 8. MMU filtering (no .And on ComputedObject) ---
    min_px = ee.Number(mmu_ha).multiply(100).round().max(1)
    
    def keep_big(c):
        c = ee.Number(c)
        mask = cls_smooth.eq(c)  # boolean image (0/1)
        valid = mask.connectedPixelCount(maxSize=1e6, eightConnected=True).gte(min_px)  # boolean
        combined = mask.multiply(valid).gt(0)  # boolean AND
        return cls_smooth.updateMask(combined)
    
    cls_mmu = ee.ImageCollection(
        ee.List.sequence(1, n_zones).map(lambda c: keep_big(ee.Number(c)))
    ).mosaic().rename("zone").toInt8().clip(geom)
    
    # --- 9. Polygonize ---
    vectors = cls_mmu.reduceToVectors(
        geometry=geom,
        scale=10,
        geometryType="polygon",
        labelProperty="zone",
        reducer=ee.Reducer.first(),
        bestEffort=True,
        maxPixels=1e9,
    )
    
    # (Optional) light debug – avoid heavy getInfo in prod
    # print("NDVI thresholds (server-side list):", thresholds)
    
    return cls_mmu, vectors

def _stream_zonal_stats(classified_path: Path, ndvi_path: Path) -> List[Dict[str, Any]]:
    stats: Dict[int, Dict[str, Any]] = {}
    with rasterio.open(classified_path) as csrc, rasterio.open(ndvi_path) as nsrc:
        assert csrc.width == nsrc.width and csrc.height == nsrc.height
        band = 1
        try:
            windows = list(csrc.block_windows(band))
        except Exception:
            step_x, step_y = 512, 512
            cols, rows = csrc.width, csrc.height
            windows = [
                ((0, 0), Window(x, y, min(step_x, cols - x), min(step_y, rows - y)))
                for y in range(0, rows, step_y)
                for x in range(0, cols, step_x)
            ]
        pixel_area = abs(csrc.transform.a) * abs(csrc.transform.e)
        for _, window in windows:
            zones = csrc.read(band, window=window)
            ndvi = nsrc.read(band, window=window, masked=True).filled(np.nan)
            mask = (zones > 0) & np.isfinite(ndvi)
            if not np.any(mask):
                continue
            zone_values = zones[mask].astype(np.int32, copy=False)
            ndvi_values = ndvi[mask].astype(np.float32, copy=False)
            for zone_id in np.unique(zone_values):
                mask_zone = zone_values == zone_id
                zone_ndvi = ndvi_values[mask_zone]
                entry = stats.setdefault(
                    int(zone_id),
                    {
                        "zone": int(zone_id),
                        "area_ha": 0.0,
                        "mean_ndvi": 0.0,
                        "min_ndvi": +1.0,
                        "max_ndvi": -1.0,
                        "pixel_count": 0,
                    },
                )
                entry["pixel_count"] += int(zone_ndvi.size)
                entry["area_ha"] += float(zone_ndvi.size * pixel_area / 10_000.0)
                entry["mean_ndvi"] += float(zone_ndvi.sum())
                entry["min_ndvi"] = float(min(entry["min_ndvi"], float(np.nanmin(zone_ndvi))))
                entry["max_ndvi"] = float(max(entry["max_ndvi"], float(np.nanmax(zone_ndvi))))
    for entry in stats.values():
        if entry["pixel_count"] > 0:
            entry["mean_ndvi"] = float(entry["mean_ndvi"] / entry["pixel_count"])
    return sorted(stats.values(), key=lambda item: item["zone"])


def _prepare_selected_period_artifacts(
    aoi_geojson: Union[dict, ee.Geometry],
    *,
    geometry: ee.Geometry,
    working_dir: Path,
    months: Sequence[str],
    start_date: date,
    end_date: date,
    cloud_prob_max: int,
    n_classes: int,
    cv_mask_threshold: float,
    apply_stability_mask: bool | None,
    min_mapping_unit_ha: float,
    smooth_radius_m: float,
    open_radius_m: float,
    close_radius_m: float,
    simplify_tol_m: float,
    simplify_buffer_m: float,
    method: str,
    sample_size: int,
    include_stats: bool,
) -> tuple[ZoneArtifacts, Dict[str, object]]:
    """Prepares classified NDVI production zones, raster + vector exports, and metadata."""
    try:
        ordered_months = _ordered_months(months)
        skipped_months: List[str] = []

        # --- 1. Validate imagery availability ---
        for month in ordered_months:
            try:
                collection, _ = gee.monthly_sentinel2_collection(geometry, month, cloud_prob_max)
                count = int(ee.Number(collection.size()).getInfo() or 0)
            except Exception:
                count = 0
            if count == 0:
                skipped_months.append(month)

        used_months = [m for m in ordered_months if m not in skipped_months]
        if not used_months:
            raise ValueError(
                "No Sentinel-2 imagery available for the selected period. "
                "Try adjusting the date range or cloud probability threshold."
            )

        # --- 2. Build mean NDVI ---
        ndvi_mean_native = _build_mean_ndvi_for_zones(
            geometry,
            start_date,
            end_date,
            months=ordered_months,
            cloud_prob_max=cloud_prob_max,
        )

        if ndvi_mean_native is None:
            raise ValueError(
                "NDVI computation failed — Earth Engine returned an empty or invalid image."
            )

        # --- 3. Validate NDVI pixel mask ---
        try:
            cnt_dict = ndvi_mean_native.mask().reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=geometry,
                scale=40,
                bestEffort=True,
                maxPixels=1e9,
            )
            cnt_value = ee.Number(ee.Dictionary(cnt_dict).values().get(0)).getInfo()
            if cnt_value is None or cnt_value <= 0:
                raise ValueError(NDVI_MASK_EMPTY_ERROR)
        except Exception as e:
            print("⚠️ Warning: NDVI pixel count check skipped due to:", str(e))
            cnt_value = 1

        if cnt_value <= 0:
            raise ValueError(NDVI_MASK_EMPTY_ERROR)

        # --- 4. Classify and polygonize zones ---
        classified_image, vectors = _classify_smooth_and_polygonize(
            ndvi_mean_native,
            geometry,
            n_zones=n_classes,
            mmu_ha=min_mapping_unit_ha,
            smooth_radius_px=DEFAULT_SMOOTH_RADIUS_PX,
        )

        if classified_image is None or vectors is None:
            raise ValueError(
                "Zone classification failed — no valid NDVI variation or geometry produced."
            )

        # --- 5. Exports ---
        workdir = _ensure_working_directory(working_dir)
        ndvi_path = workdir / "NDVI_mean.tif"
        mean_export = _download_image_to_path(
            ndvi_mean_native,
            geometry,
            ndvi_path,
            params=_DownloadParams(crs=DEFAULT_EXPORT_CRS, scale=DEFAULT_SCALE),
        )
        ndvi_path = mean_export.path

        classified_path = workdir / "zones.tif"
        classified_export = _download_image_to_path(
            classified_image,
            geometry,
            classified_path,
            params=_DownloadParams(crs=DEFAULT_EXPORT_CRS, scale=DEFAULT_SCALE),
        )
        classified_path = classified_export.path

        # --- 6. Identify unique classes ---
        try:
            unique_classes = (
                classified_image.reduceRegion(
                    reducer=ee.Reducer.frequencyHistogram(),
                    geometry=geometry,
                    scale=DEFAULT_SCALE,
                    bestEffort=True,
                    tileScale=4,
                    maxPixels=gee.MAX_PIXELS,
                ).get("zone")
            )
            unique_classes = (
                list((unique_classes or {}).keys())
                if isinstance(unique_classes, dict)
                else []
            )
        except Exception:
            unique_classes = []

        if not unique_classes:
            raise ValueError(
                "NDVI variation too low to produce multiple zones. "
                "Try a wider time range or larger AOI."
            )

        # --- 7. Reproject + vector exports ---
        vectors_reprojected = ee.FeatureCollection(
            vectors.map(
                lambda f: f.setGeometry(f.geometry().transform("EPSG:4326", 0.1))
            )
        )

        geojson_path = workdir / "zones.geojson"
        kml_path = workdir / "zones.kml"
        _download_vector_to_path(vectors_reprojected, geojson_path, file_format="geojson")
        _download_vector_to_path(vectors_reprojected, kml_path, file_format="kml")

        # --- 8. Optional zonal stats ---
        stats_path = None
        zonal_stats: List[Dict[str, Any]] = []
        if include_stats:
            stats_path = workdir / "zones_zonal_stats.csv"
            zonal_stats = _stream_zonal_stats(classified_path, ndvi_path)
            with stats_path.open("w", newline="") as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=[
                        "zone",
                        "area_ha",
                        "mean_ndvi",
                        "min_ndvi",
                        "max_ndvi",
                        "pixel_count",
                    ],
                )
                writer.writeheader()
                writer.writerows(zonal_stats)

        # --- 9. Metadata ---
        palette = list(ZONE_PALETTE[: max(1, min(n_classes, len(ZONE_PALETTE)))])
        metadata: Dict[str, Any] = {
            "used_months": used_months,
            "skipped_months": skipped_months,
            "zone_method": "ndvi_quantiles",
            "stability_mask_applied": False,
            "percentile_thresholds": [],
            "palette": palette,
            "requested_zone_count": int(n_classes),
            "effective_zone_count": int(len(unique_classes) or n_classes),
            "zones": zonal_stats,
            "classification_method": "quantiles",
        }

        artifacts = ZoneArtifacts(
            raster_path=str(classified_path),
            mean_ndvi_path=str(ndvi_path),
            vector_path=str(geojson_path),
            vector_components={
                "geojson": str(geojson_path),
                "kml": str(kml_path),
            },
            zonal_stats_path=str(stats_path) if stats_path else None,
            working_dir=str(workdir),
        )

        metadata["downloaded_mean_ndvi"] = str(ndvi_path)
        metadata["mean_ndvi_export_task"] = _task_payload(mean_export.task)
        metadata["classified_export_task"] = _task_payload(classified_export.task)
        return artifacts, metadata

    # --- Controlled EE exception translation ---
    except ee.ee_exception.EEException as ee_err:
        msg = str(ee_err)
        if "Image.constant" in msg or "may not be null" in msg:
            raise ValueError(
                "NDVI computation failed — empty or invalid image returned from Earth Engine. "
                "Try widening the date range or relaxing cloud masking."
            )
        elif "Collection" in msg and "empty" in msg:
            raise ValueError(
                "No Sentinel-2 imagery found for the selected period."
            )
        else:
            raise ValueError(f"Earth Engine error: {msg}")

    # --- Preserve other ValueErrors (already user-facing) ---
    except ValueError:
        raise

    # --- Catch-all fallback ---
    except Exception as exc:
        raise ValueError(f"Unexpected error while preparing NDVI zones: {exc}")

def build_zone_artifacts(
    aoi_geojson: Union[dict, ee.Geometry],
    *,
    months: Sequence[str],
    cloud_prob_max: int = DEFAULT_CLOUD_PROB_MAX,
    n_classes: int = DEFAULT_N_CLASSES,
    cv_mask_threshold: float = DEFAULT_CV_THRESHOLD,
    apply_stability_mask: bool | None = None,
    min_mapping_unit_ha: float = DEFAULT_MIN_MAPPING_UNIT_HA,
    smooth_radius_m: float = 0,
    open_radius_m: float = 0,
    close_radius_m: float = 0,
    simplify_tolerance_m: float = 5,
    simplify_buffer_m: float = 3,
    method: str = DEFAULT_METHOD,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    include_stats: bool = True,
) -> ZoneArtifacts:
    if n_classes < 2 or n_classes > 7:
        raise ValueError("n_classes must be between 2 and 7")
    if not months:
        raise ValueError("At least one month must be supplied")

    method_key = (method or "").strip().lower() or DEFAULT_METHOD
    if method_key != "ndvi_percentiles":
        logger.warning("Unsupported method %s requested; falling back to ndvi_percentiles", method_key)
        method_key = "ndvi_percentiles"

    try:
        gee.initialize()
    except Exception:
        if not _allow_init_failure():
            raise

    geometry = _resolve_geometry(aoi_geojson)
    start_date, end_date = _month_range_dates(months)
    working_dir = _ensure_working_directory(None)

    artifacts, _metadata = _prepare_selected_period_artifacts(
        aoi_geojson,
        geometry=geometry,
        working_dir=working_dir,
        months=months,
        start_date=start_date,
        end_date=end_date,
        cloud_prob_max=cloud_prob_max,
        n_classes=n_classes,
        cv_mask_threshold=cv_mask_threshold,
        apply_stability_mask=apply_stability_mask,
        min_mapping_unit_ha=min_mapping_unit_ha,
        smooth_radius_m=smooth_radius_m,
        open_radius_m=open_radius_m,
        close_radius_m=close_radius_m,
        simplify_tol_m=simplify_tolerance_m,
        simplify_buffer_m=simplify_buffer_m,
        method=method_key,
        sample_size=sample_size,
        include_stats=include_stats,
    )
    return artifacts


def export_selected_period_zones(
    aoi_geojson: dict,
    aoi_name: str,
    months: list[str],
    *,
    geometry: ee.Geometry | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    cloud_prob_max: int = 80,
    n_classes: int = 5,
    cv_mask_threshold: float | None = None,
    mmu_ha: float = 2.0,
    min_mapping_unit_ha: float | None = None,
    smooth_radius_m: int = 0,
    open_radius_m: int = 0,
    close_radius_m: int = 0,
    simplify_tol_m: int = 5,
    simplify_tolerance_m: int | None = None,
    simplify_buffer_m: int = 3,
    export_target: str = "local",
    destination: str | None = None,
    gcs_bucket: str | None = None,
    gcs_prefix: str | None = None,
    include_zonal_stats: bool = True,
    include_stats: bool | None = None,
    apply_stability_mask: bool = True,
    method: str | None = None,
) -> Dict[str, Any]:
    working_dir = _ensure_working_directory(None)
    aoi = _to_ee_geometry(aoi_geojson)
    geometry = geometry or aoi

    if start_date is not None and end_date is not None and end_date < start_date:
        raise ValueError("end_date must be on or after start_date")

    if cv_mask_threshold is None:
        cv_mask_threshold = DEFAULT_CV_THRESHOLD
    if min_mapping_unit_ha is not None:
        mmu_ha = float(min_mapping_unit_ha)
    if simplify_tolerance_m is not None:
        simplify_tol_m = int(simplify_tolerance_m)
    if destination is not None:
        export_target = destination

    def _coerce_date_any(d):
        """Accepts str, date, or datetime."""
        if isinstance(d, datetime):
            return d.date()
        if isinstance(d, date):
            return d
        if isinstance(d, str):
            # handle full ISO or YYYY-MM-DD
            return datetime.fromisoformat(d[:10]).date()
        raise TypeError(f"Invalid date type: {type(d)}")


    if not months:
        if start_date is None or end_date is None:
            raise ValueError("Either months or start/end dates must be supplied")
        start_dt = _coerce_date_any(start_date)
        end_dt = _coerce_date_any(end_date)
        months = _months_from_dates(start_dt, end_dt)
    
    ordered_months = _ordered_months(months)
    
    if start_date is None or end_date is None:
        start_dt, end_dt = _month_range_dates(ordered_months)
    else:
        start_dt = _coerce_date_any(start_date)
        end_dt = _coerce_date_any(end_date)

    include_stats_flag = bool(include_stats if include_stats is not None else include_zonal_stats)

    try:
        gee.initialize()
    except Exception:
        if not _allow_init_failure():
            raise

    geometry = geometry or _resolve_geometry(aoi_geojson)

    method_key = (method or DEFAULT_METHOD).strip().lower()
    if method_key != "ndvi_percentiles":
        logger.warning("Unsupported method %s requested; falling back to ndvi_percentiles", method_key)
        method_key = "ndvi_percentiles"

    artifacts, metadata = _prepare_selected_period_artifacts(
        aoi_geojson,
        geometry=geometry,
        working_dir=working_dir,
        months=ordered_months,
        start_date=start_dt,
        end_date=end_dt,
        cloud_prob_max=cloud_prob_max,
        n_classes=n_classes,
        cv_mask_threshold=cv_mask_threshold,
        apply_stability_mask=apply_stability_mask,
        min_mapping_unit_ha=mmu_ha,
        smooth_radius_m=smooth_radius_m,
        open_radius_m=open_radius_m,
        close_radius_m=close_radius_m,
        simplify_tol_m=simplify_tol_m,
        simplify_buffer_m=simplify_buffer_m,
        method=method_key,
        sample_size=DEFAULT_SAMPLE_SIZE,
        include_stats=include_stats_flag,
    )

    metadata = dict(metadata)
    metadata["zone_method"] = method_key
    metadata = sanitize_for_json(metadata)
    used_months: list[str] = list(metadata.get("used_months", []))
    skipped: list[str] = list(metadata.get("skipped_months", []))
    if not used_months:
        raise ValueError("No valid Sentinel-2 scenes available for the selected period")

    prefix_base = _export_prefix(aoi_name, used_months)
    result: Dict[str, Any] = {
        "paths": {
            "raster": artifacts.raster_path,
            "mean_ndvi": artifacts.mean_ndvi_path,
            "vectors": artifacts.vector_path,
            "vector_components": artifacts.vector_components,
            "zonal_stats": artifacts.zonal_stats_path if include_stats_flag else None,
        },
        "tasks": {},
        "prefix": prefix_base,
        "metadata": metadata,
        "artifacts": artifacts,
        "working_dir": artifacts.working_dir or str(working_dir),
    }

    palette = metadata.get("palette")
    thresholds = metadata.get("percentile_thresholds")
    if palette is not None:
        result["palette"] = palette
    if thresholds is not None:
        result["thresholds"] = thresholds

    export_target = (export_target or "zip").strip().lower()
    if export_target not in {"zip", "local"}:
        raise ValueError("Only local zone exports are supported in this workflow")

    return result


def _task_payload(task: ee.batch.Task | None) -> Dict[str, object]:
    if task is None:
        return {}
    payload: Dict[str, object] = {"id": getattr(task, "id", None)}
    try:
        status = task.status() or {}
    except Exception:  # pragma: no cover - Earth Engine failure mode
        status = {}
    if status.get("state"):
        payload["state"] = status.get("state")
    destination = status.get("destination_uris") or []
    if destination:
        payload["destination_uris"] = destination
    error = status.get("error_message") or status.get("error_details")
    if error:
        payload["error"] = error
    return payload
