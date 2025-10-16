from __future__ import annotations

import calendar
import csv
import logging
import math
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
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
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
DEFAULT_EXPORT_CRS: str | None = None
DEFAULT_CRS = DEFAULT_EXPORT_CRS
DEFAULT_NODATA_VALUE = -9999.0

# NEW: zoning controls
DEFAULT_MODE = "linear"           # "linear" | "quantile" | "auto"
DEFAULT_NDVI_MIN = 0.35
DEFAULT_NDVI_MAX = 0.73

MIN_VALID_PIXEL_RATIO = 0.01

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
    nodata_value: float | None = DEFAULT_NODATA_VALUE


def _resolve_image_crs(image: ee.Image) -> str | None:
    projection = getattr(image, "projection", None)
    if not callable(projection):
        return None
    try:
        proj = projection()
    except Exception:
        return None

    try:
        crs_attr = getattr(proj, "crs", None)
        if callable(crs_attr):
            crs_value = crs_attr()
            info = getattr(crs_value, "getInfo", None)
            if callable(info):
                resolved = info()
                if isinstance(resolved, str) and resolved:
                    return resolved
    except Exception:
        pass

    try:
        info_method = getattr(proj, "getInfo", None)
        if callable(info_method):
            info = info_method() or {}
            crs_from_info = info.get("crs")
            if isinstance(crs_from_info, str) and crs_from_info:
                return crs_from_info
    except Exception:
        pass

    return None


def _maybe_reproject_image(
    path: Path,
    *,
    native_crs: str | None,
    output_crs: str | None,
    nodata_value: float | None = DEFAULT_NODATA_VALUE,
) -> None:
    desired_crs = output_crs or native_crs
    try:
        with rasterio.open(path) as src:
            raw_meta = getattr(src, "meta", None)
            if raw_meta is None:
                raw_meta = getattr(src, "profile", None)
            if raw_meta is None and hasattr(src, "profile_dict"):
                raw_meta = src.profile_dict
            src_meta = dict(raw_meta or {})
            profile_defaults = getattr(src, "_profile", None)
            if profile_defaults is not None:
                src_meta.setdefault("dtype", getattr(profile_defaults, "dtype", None))
                src_meta.setdefault("width", getattr(profile_defaults, "width", None))
                src_meta.setdefault("height", getattr(profile_defaults, "height", None))
                src_meta.setdefault("count", getattr(profile_defaults, "count", None))
                src_meta.setdefault("transform", getattr(profile_defaults, "transform", None))
                src_meta.setdefault("crs", getattr(profile_defaults, "crs", None))
                src_meta.setdefault("nodata", getattr(profile_defaults, "nodata", None))
            src_meta.setdefault("dtype", getattr(src_meta.get("dtype"), "name", src_meta.get("dtype")))
            src_meta.setdefault("width", getattr(src, "width", None))
            src_meta.setdefault("height", getattr(src, "height", None))
            src_meta.setdefault("count", getattr(src, "count", None))
            src_meta.setdefault("transform", getattr(src, "transform", None))
            src_meta.setdefault("crs", getattr(src, "crs", None))
            src_meta.setdefault("nodata", getattr(src, "nodata", None))
            if src_meta.get("dtype") is None:
                src_meta["dtype"] = np.float32
            src_crs = src.crs
            src_nodata = src_meta.get("nodata")
            src_dtype = np.dtype(src_meta["dtype"])

            dst_crs = None
            if desired_crs:
                try:
                    dst_crs = rasterio.crs.CRS.from_user_input(desired_crs)
                except Exception:
                    dst_crs = None

            needs_reproject = bool(dst_crs and src_crs and dst_crs != src_crs)
            needs_assign_crs = bool(dst_crs and src_crs is None)
            needs_dtype_update = not np.issubdtype(src_dtype, np.floating)
            target_nodata = (
                nodata_value if nodata_value is not None else src_nodata
            )
            if target_nodata is None or not np.isfinite(target_nodata):
                target_nodata = DEFAULT_NODATA_VALUE

            if src_nodata is not None and np.isfinite(src_nodata):
                sanitized_src_nodata = float(src_nodata)
            else:
                sanitized_src_nodata = None

            needs_nodata_update = (
                sanitized_src_nodata is None or sanitized_src_nodata != target_nodata
            )

            if not any((needs_reproject, needs_assign_crs, needs_dtype_update, needs_nodata_update)):
                return

            dst_transform = src.transform
            dst_width, dst_height = src.width, src.height
            if needs_reproject and dst_crs:
                dst_transform, dst_width, dst_height = calculate_default_transform(
                    src_crs,
                    dst_crs,
                    src.width,
                    src.height,
                    *src.bounds,
                )

            target_crs = dst_crs or src_crs
            target_dtype = "float32" if needs_dtype_update else src_meta["dtype"]
            effective_src_nodata = target_nodata

            dst_meta = src_meta.copy()
            dst_meta.update(
                crs=target_crs,
                transform=dst_transform,
                width=dst_width,
                height=dst_height,
                nodata=target_nodata,
                dtype=target_dtype,
            )

            fd, tmp_name = tempfile.mkstemp(suffix=".tif")
            os.close(fd)
            tmp_path = Path(tmp_name)
            try:
                resampling = (
                    Resampling.bilinear if np.issubdtype(src_dtype, np.floating) else Resampling.nearest
                )
                with rasterio.open(tmp_path, "w", **dst_meta) as dst:
                    dst_dtype = np.dtype(dst_meta["dtype"])
                    for band_idx in range(1, src.count + 1):
                        source_band = src.read(band_idx, masked=True)
                        if isinstance(source_band, np.ma.MaskedArray):
                            source_array = source_band.filled(target_nodata)
                        else:
                            source_array = source_band
                        source_array = np.asarray(source_array, dtype=dst_dtype)
                        try:
                            reproject(
                                source=source_array,
                                destination=rasterio.band(dst, band_idx),
                                src_transform=src.transform,
                                src_crs=src_crs or target_crs,
                                dst_transform=dst_transform,
                                dst_crs=target_crs,
                                src_nodata=effective_src_nodata,
                                dst_nodata=target_nodata,
                                resampling=resampling,
                            )
                        except TypeError:
                            tmp_src_path: Path | None = None
                            try:
                                fd, tmp_name_src = tempfile.mkstemp(suffix=".tif")
                                os.close(fd)
                                tmp_src_path = Path(tmp_name_src)
                                tmp_meta = dict(
                                    driver="GTiff",
                                    height=src.height,
                                    width=src.width,
                                    count=1,
                                    dtype=dst_dtype,
                                    crs=src_crs or target_crs,
                                    transform=src.transform,
                                    nodata=target_nodata,
                                )
                                with rasterio.open(tmp_src_path, "w", **tmp_meta) as tmp_src_ds:
                                    reshaped = source_array.reshape(1, source_array.shape[0], source_array.shape[1])
                                    tmp_src_ds.write(reshaped, 1)
                                with rasterio.open(tmp_src_path) as tmp_src_ds:
                                    reproject(
                                        source=rasterio.band(tmp_src_ds, 1),
                                        destination=rasterio.band(dst, band_idx),
                                        src_transform=src.transform,
                                        src_crs=src_crs or target_crs,
                                        dst_transform=dst_transform,
                                        dst_crs=target_crs,
                                        src_nodata=effective_src_nodata,
                                        dst_nodata=target_nodata,
                                        resampling=resampling,
                                    )
                            finally:
                                if tmp_src_path is not None:
                                    try:
                                        tmp_src_path.unlink(missing_ok=True)
                                    except Exception:
                                        pass
                    if target_nodata is not None:
                        for band_idx in range(1, dst.count + 1):
                            data = dst.read(band_idx, masked=True)
                            if isinstance(data, np.ma.MaskedArray) and data.mask is not np.ma.nomask:
                                if np.any(data.mask):
                                    dst.write(data.filled(target_nodata), band_idx)
                tmp_path.replace(path)
            finally:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
    except Exception:
        logger.exception("Failed to post-process downloaded GeoTIFF at %s", path)


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

    native_crs = _resolve_image_crs(image)

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

    _maybe_reproject_image(
        target,
        native_crs=native_crs,
        output_crs=params.crs,
        nodata_value=params.nodata_value,
    )
    return ImageExportResult(path=target, task=task)



def _ensure_minimum_valid_pixels(
    path: Path,
    *,
    label: str,
    threshold: float = MIN_VALID_PIXEL_RATIO,
) -> None:
    if threshold <= 0:
        return

    with rasterio.open(path) as src:
        band = src.read(1, masked=True)
        data = np.asarray(band)
        total_pixels = int(band.size)

        if isinstance(band, np.ma.MaskedArray):
            mask = np.array(band.mask, dtype=bool)
            if mask.shape == ():
                mask = np.full(data.shape, bool(mask))
        else:
            mask = np.zeros(data.shape, dtype=bool)

        nodata_values: set[float] = set()
        nodata = getattr(src, "nodata", None)
        if nodata is not None:
            try:
                nodata_float = float(nodata)
            except (TypeError, ValueError):
                nodata_float = None
            if nodata_float is not None and not math.isnan(nodata_float):
                nodata_values.add(nodata_float)
        nodatavals = getattr(src, "nodatavals", None)
        if nodatavals:
            for candidate in nodatavals:
                if candidate is None:
                    continue
                try:
                    candidate_float = float(candidate)
                except (TypeError, ValueError):
                    continue
                if math.isnan(candidate_float):
                    continue
                nodata_values.add(candidate_float)

        if nodata_values:
            comparison = np.zeros(data.shape, dtype=bool)
            for value in nodata_values:
                comparison |= np.isclose(data, value)
            mask |= comparison

        invalid_pixels = int(np.count_nonzero(mask)) if total_pixels else total_pixels
        valid_pixels = total_pixels - invalid_pixels
        valid_ratio = (valid_pixels / total_pixels) if total_pixels else 0.0

    if valid_ratio < threshold:
        raise ValueError(
            f"Downloaded {label} GeoTIFF contains fewer than {threshold:.1%} valid pixels. "
            "Revisit the mask configuration or selected date range."
        )





def _download_vector_to_path(vectors: ee.FeatureCollection, target: Path, *, file_format: str) -> None:
    fmt = (file_format or "geojson").strip().lower()
    ft_map = {
        "geojson": "GEO_JSON", "json": "GEO_JSON", "kml": "KML",
        "csv": "CSV", "shp": "SHP", "shapefile": "SHP",
    }
    filetype = ft_map.get(fmt, "GEO_JSON")
    vectors = ee.FeatureCollection(vectors.map(lambda f: f.setGeometry(f.geometry().transform("EPSG:4326", 0.1))))
    if fmt == "geojson" and target.suffix.lower() != ".geojson":
        target = target.with_suffix(".geojson")
    elif fmt == "kml" and target.suffix.lower() != ".kml":
        target = target.with_suffix(".kml")
    elif fmt == "csv" and target.suffix.lower() != ".csv":
        target = target.with_suffix(".csv")
    elif fmt in {"shp", "shapefile"} and target.suffix.lower() != ".zip":
        target = target.with_suffix(".zip")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        url = vectors.getDownloadURL(filetype=filetype, selectors=["zone"], filename=target.stem)
    except Exception as e:
        raise ValueError(f"EE getDownloadURL failed: {e}")
    from urllib.request import urlopen
    with urlopen(url) as r, open(target, "wb") as f:
        f.write(r.read())


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


# ---------------------------------------------------------------------------
# NDVI MEAN BUILDER (robust mask; warn on low variance; native reprojection)
# ---------------------------------------------------------------------------
def _build_mean_ndvi_for_zones(
    geom: ee.Geometry,
    start_date: date | str,
    end_date: date | str,
    *,
    months: Sequence[str] | None = None,
    cloud_prob_max: int = 60,
) -> ee.Image:
    """Build mean NDVI for the selected period (warn on low variation, don't abort)."""

    def _iso(d):
        return d.isoformat() if isinstance(d, date) else str(d)

    def _to_date(value: date | str) -> date:
        if isinstance(value, date):
            return value
        return datetime.fromisoformat(str(value)[:10]).date()

    if months:
        ordered_months = _ordered_months(months)
        start_iso = _iso(start_date)
        end_iso = _iso(end_date)
    else:
        start_dt = _to_date(start_date)
        end_dt = _to_date(end_date)
        ordered_months = _months_from_dates(start_dt, end_dt)
        start_iso = start_dt.isoformat()
        end_iso = end_dt.isoformat()

    ndvi_collections: List[ee.ImageCollection] = []
    first_reference: ee.Image | None = None

    def _ndvi(img: ee.Image) -> ee.Image:
        """
        Compute NDVI (or reuse NDVI_mean) with a single-band mask (B8 ∧ B4).
        Avoids multi-band img.mask() side effects and NaN constants.
        """
        bnames = ee.List(img.bandNames())
        has_precomputed = ee.Number(bnames.indexOf("NDVI_mean")).gte(0)
        ndvi_from_band = ee.Image(
            ee.Algorithms.If(
                has_precomputed,
                img.select("NDVI_mean").rename("NDVI"),
                ee.Image(0).updateMask(ee.Image(0)).rename("NDVI"),
            )
        )

        has_b4 = ee.Number(bnames.indexOf("B4")).gte(0)
        has_b8 = ee.Number(bnames.indexOf("B8")).gte(0)
        both = ee.Number(has_b4).multiply(ee.Number(has_b8)).eq(1)

        computed_ndvi = img.toFloat().normalizedDifference(["B8", "B4"]).rename("NDVI")
        ndvi = ee.Image(
            ee.Algorithms.If(
                has_precomputed,
                ndvi_from_band,
                ee.Algorithms.If(both, computed_ndvi, ndvi_from_band),
            )
        )

        mask_1band = img.select("B8").mask().multiply(img.select("B4").mask()).gt(0)
        return ee.Image(ndvi).updateMask(mask_1band)

    for month_str in ordered_months:
        month_result = gee.monthly_sentinel2_collection(
            geom,
            month_str,
            cloud_prob_max=cloud_prob_max,
        )
        if isinstance(month_result, tuple):
            month_collection = month_result[0]
        else:
            month_collection = month_result

        try:
            month_size = int(ee.Number(month_collection.size()).getInfo() or 0)
        except Exception:
            month_size = 0
        if month_size == 0:
            continue

        ndvi_collections.append(month_collection.map(_ndvi))
        if first_reference is None:
            first_reference = ee.Image(month_collection.first())

    if not ndvi_collections:
        raise ValueError(S2_COLLECTION_EMPTY_ERROR)

    ndvi_collection = ndvi_collections[0]
    for extra in ndvi_collections[1:]:
        ndvi_collection = ndvi_collection.merge(extra)

    # Keep NDVI masked for stats (avoid unmask(0) before thresholds)
    ndvi_mean = mean_from_collection_sum_count(ndvi_collection).toFloat().clip(geom)
    ndvi_mean = ndvi_mean.set({"period_start": start_iso, "period_end": end_iso})

    # 3) Variation check → warn, don't raise
    try:
        std_dict = ee.Dictionary(
            ndvi_mean.reduceRegion(
                reducer=ee.Reducer.stdDev(),
                geometry=geom,
                scale=10,
                bestEffort=True,
                maxPixels=1e9,
            )
        )
        std_val = ee.Number(
            ee.Algorithms.If(
                std_dict.contains("NDVI_mean_stdDev"),
                std_dict.get("NDVI_mean_stdDev"),
                ee.Dictionary(
                    ndvi_mean.reduceRegion(
                        reducer=ee.Reducer.stdDev(),
                        geometry=geom,
                        scale=10,
                        bestEffort=True,
                        maxPixels=1e9,
                    )
                ).values().get(0),
            )
        ).getInfo()
    except Exception:
        std_val = 0.0

    if std_val is None:
        std_val = 0.0

    if std_val < 1e-3:
        logger.warning("NDVI variation low (std=%.6f). Proceeding; zones may collapse.", std_val)
        ndvi_mean = ndvi_mean.set({"ndvi_stdDev": std_val, "ndvi_low_variation": True})
    else:
        ndvi_mean = ndvi_mean.set({"ndvi_stdDev": std_val, "ndvi_low_variation": False})

    # 4) Native 10 m reprojection if reference is available
    first_img = ee.Image(first_reference) if first_reference is not None else ee.Image(ndvi_collection.first())
    ndvi_native = ee.Algorithms.If(
        first_img,
        reproject_native_10m(ndvi_mean, first_img, ref_band="B8", scale=10),
        ndvi_mean,
    )

    return ee.Image(ndvi_native).rename("NDVI_mean").toFloat().clip(geom)


# ---------------------------------------------------------------------------
# CLASSIFY + OPTIONAL SMOOTH + POLYGONIZE
# ---------------------------------------------------------------------------
def _classify_smooth_and_polygonize(
    ndvi_mean_native: ee.Image,
    geom: ee.Geometry,
    *,
    n_zones: int = 3,
    mmu_ha: float = 1.0,
    smooth_radius_px: int = 1,
    mode: str = DEFAULT_MODE,                 # "linear" | "quantile" | "auto"
    ndvi_min: float | None = None,
    ndvi_max: float | None = None,
):
    """
    Classify NDVI into production zones.

    Modes:
      - 'linear': fixed equal-width bins in [ndvi_min, ndvi_max] (defaults 0.35–0.73)
      - 'quantile': equal-frequency bins (by percentiles)
      - 'auto': use quantile if variation exists, else linear
    """
    ndvi = ndvi_mean_native.rename("NDVI_mean").clip(geom)

    lo = DEFAULT_NDVI_MIN if ndvi_min is None else float(ndvi_min)
    hi = DEFAULT_NDVI_MAX if ndvi_max is None else float(ndvi_max)
    if hi <= lo:
        hi = lo + 1e-6  # avoid zero span

    # --- detect flatness from image metadata (server -> client safely) ---
    try:
        flat_prop = ndvi.get("ndvi_low_variation")
        flat_flag = str(flat_prop.getInfo()).lower()
    except Exception:
        flat_flag = "false"

    # --- thresholds selection ---
    if mode == "quantile" or (mode == "auto" and flat_flag != "true"):
        # quantile bin edges (0..100)
        percentiles = [100.0 * (i / n_zones) for i in range(n_zones + 1)]
        stats = ndvi.reduceRegion(
            reducer=ee.Reducer.percentile(percentiles),
            geometry=geom,
            scale=10,
            bestEffort=True,
            maxPixels=1e9,
        )
        thresholds_py: List[float] = []
        for p in percentiles:
            key = f"NDVI_mean_p{int(p)}"
            try:
                val = ee.Number(ee.Dictionary(stats).get(key)).getInfo()
                if val is None:
                    raise Exception()
                thresholds_py.append(float(val))
            except Exception:
                # fallback to linear edge inside [lo, hi]
                thresholds_py.append(lo + (hi - lo) * (p / 100.0))
        method_used = "quantile"
    else:
        # linear (default and 'auto' fallback when flat)
        step = (hi - lo) / n_zones
        thresholds_py = [lo + i * step for i in range(n_zones + 1)]
        method_used = "linear"

    # --- classification by thresholds ---
    def classify(img: ee.Image, edges: List[float]) -> ee.Image:
        zones = ee.Image.constant(0)
        for i in range(len(edges) - 1):
            lower = edges[i]
            upper = edges[i + 1]
            zones = zones.where(img.gte(lower).And(img.lt(upper)), i + 1)
        # include the top edge in last bin
        zones = zones.where(img.gte(edges[-2]).And(img.lte(edges[-1])), len(edges) - 1)
        return zones.rename("zone").updateMask(img.mask()).toInt8()

    cls_raw = classify(ndvi, thresholds_py)

    # --- optional smoothing ---
    cls_smooth = ee.Image(
        ee.Algorithms.If(
            ee.Number(smooth_radius_px).gt(0),
            cls_raw.focalMode(radius=smooth_radius_px, units="pixels"),
            cls_raw,
        )
    ).toInt8()

    # --- MMU filter ---
    min_px = ee.Number(mmu_ha).multiply(100).round().max(1)

    def keep_big(c):
        mask = cls_smooth.eq(c)
        # EE limit for maxSize is 1024; omit it for full-tile connectivity
        valid = mask.connectedPixelCount(eightConnected=True).gte(min_px)
        return cls_smooth.updateMask(mask.And(valid))

    cls_mmu = ee.ImageCollection(
        [keep_big(ee.Number(c)) for c in range(1, n_zones + 1)]
    ).mosaic().rename("zone").toInt8().clip(geom)

    # --- vectorize ---
    vectors = cls_mmu.reduceToVectors(
        geometry=geom,
        scale=10,
        geometryType="polygon",
        labelProperty="zone",
        reducer=ee.Reducer.countEvery(),
        bestEffort=True,
        maxPixels=1e9,
    )

    # metadata on the image
    cls_mmu = cls_mmu.set(
        {
            "thresholds": thresholds_py,
            "zones": n_zones,
            "mode": method_used,
            "ndvi_min": lo,
            "ndvi_max": hi,
        }
    )

    return cls_mmu, vectors


def _stream_zonal_stats(classified_path: Path, ndvi_path: Path) -> List[Dict[str, Any]]:
    stats: Dict[int, Dict[str, Any]] = {}
    with rasterio.open(classified_path) as csrc, rasterio.open(ndvi_path) as nsrc:
        assert csrc.width == nsrc.width and csrc.height == nsrc.height
        band = 1
        nodata_value = None
        try:
            nodata_values = getattr(nsrc, "nodatavals", None)
            if nodata_values:
                nodata_value = nodata_values[band - 1]
        except Exception:  # pragma: no cover - defensive; nodatavals can misbehave
            nodata_value = None
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
            ndvi = nsrc.read(band, window=window).astype(np.float32, copy=False)
            mask_values = nsrc.read_masks(band, window=window)
            mask = zones > 0
            if mask_values is not None:
                mask &= mask_values.astype(bool)
            mask &= np.isfinite(ndvi)
            if nodata_value is not None and not np.isnan(nodata_value):
                mask &= ndvi != float(nodata_value)
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
    # NEW:
    mode: str = DEFAULT_MODE,
    ndvi_min: float | None = None,
    ndvi_max: float | None = None,
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
        
        ndvi_mean_export = (
            ee.Image(ndvi_mean_native)
            .rename('NDVI_mean')
            .toFloat()
            .unmask(-9999)
            .clip(geometry)
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
                scale=10,
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
            smooth_radius_px=max(0, int(round(smooth_radius_m / 10))),
            mode=mode,
            ndvi_min=ndvi_min,
            ndvi_max=ndvi_max,
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
            params=_DownloadParams(scale=DEFAULT_SCALE, nodata_value=DEFAULT_NODATA_VALUE),
        )
        _ensure_minimum_valid_pixels(mean_export.path, label="NDVI mean")
        ndvi_path = mean_export.path

        classified_path = workdir / "zones.tif"
        classified_export = _download_image_to_path(
            classified_image,
            geometry,
            classified_path,
            params=_DownloadParams(scale=DEFAULT_SCALE, nodata_value=0),
        )
        _ensure_minimum_valid_pixels(classified_export.path, label="classified zones")
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

        if not unique_classes or len(unique_classes) < 2:
            logger.warning("NDVI variation very low — forcing linear fallback zones.")
            # force 3-band linear bins
            classified_image, vectors = _classify_smooth_and_polygonize(
                ndvi_mean_native,
                geometry,
                n_zones=max(3, n_classes),
                mmu_ha=min_mapping_unit_ha,
                smooth_radius_px=max(0, int(round(smooth_radius_m / 10))),
                mode="linear",
                ndvi_min=0.2,
                ndvi_max=0.9,
            )

        # --- 7. Reproject + vector exports ---
        vectors_reprojected = ee.FeatureCollection(
            vectors.map(lambda f: f.setGeometry(f.geometry().transform("EPSG:4326", 0.1)))
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
            "stability_mask_applied": False,
            "palette": palette,
            "requested_zone_count": int(n_classes),
            "effective_zone_count": int(len(unique_classes) or n_classes),
            "zones": zonal_stats,
            # classification info:
            "classification_mode": classified_image.get("mode").getInfo() if hasattr(classified_image, "get") else mode,
            "percentile_thresholds": [],
            "thresholds": classified_image.get("thresholds").getInfo() if hasattr(classified_image, "get") else [],
            "ndvi_min": classified_image.get("ndvi_min").getInfo() if hasattr(classified_image, "get") else ndvi_min,
            "ndvi_max": classified_image.get("ndvi_max").getInfo() if hasattr(classified_image, "get") else ndvi_max,
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
            raise ValueError("No Sentinel-2 imagery found for the selected period.")
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
    # NEW:
    mode: str = DEFAULT_MODE,
    ndvi_min: float | None = None,
    ndvi_max: float | None = None,
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
        mode=mode,
        ndvi_min=ndvi_min,
        ndvi_max=ndvi_max,
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
    # NEW:
    mode: str = DEFAULT_MODE,
    ndvi_min: float | None = None,
    ndvi_max: float | None = None,
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
        mode=mode,                 # NEW
        ndvi_min=ndvi_min,         # NEW
        ndvi_max=ndvi_max,         # NEW
    )

    metadata = dict(metadata)
    metadata["zone_method"] = method_key
    metadata = sanitize_for_json(metadata)
    used_months: list[str] = list(metadata.get("used_months", []))
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
    thresholds = metadata.get("thresholds") or metadata.get("percentile_thresholds")
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
