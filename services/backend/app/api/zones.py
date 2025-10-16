
from __future__ import annotations

import calendar
import csv
import json
import logging
import os
import secrets
import shutil
import tempfile
import threading
import time
import zipfile
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
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator, validator
from rasterio.windows import Window
from starlette.background import BackgroundTask

from app import gee
from app.exports import sanitize_name
from app.api.s2_indices import router as router
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
DEFAULT_MODE = "linear"  # "linear" | "quantile" | "auto"
DEFAULT_SAMPLE_SIZE = 4000
DEFAULT_SCALE = int(os.getenv("ZONES_SCALE_M", "10"))
DEFAULT_EXPORT_CRS = "EPSG:3857"
DEFAULT_CRS = DEFAULT_EXPORT_CRS

ZONE_BUNDLE_TTL_SECONDS = 60 * 60  # 1 hour


class ProductionZonesRequest(BaseModel):
    aoi_geojson: dict
    aoi_name: str
    months: List[str] = Field(default_factory=list)
    start_month: str | None = Field(None, description="YYYY-MM")
    end_month: str | None = Field(None, description="YYYY-MM")
    cloud_prob_max: int = Field(DEFAULT_CLOUD_PROB_MAX, ge=0, le=100)
    n_classes: int = Field(DEFAULT_N_CLASSES, ge=2, le=7)
    cv_mask_threshold: float = Field(DEFAULT_CV_THRESHOLD, ge=0)
    apply_stability_mask: bool = True
    mmu_ha: float = Field(DEFAULT_MIN_MAPPING_UNIT_HA, gt=0)
    smooth_radius_m: float = Field(DEFAULT_SMOOTH_RADIUS_M, ge=0)
    open_radius_m: float = Field(DEFAULT_OPEN_RADIUS_M, ge=0)
    close_radius_m: float = Field(DEFAULT_CLOSE_RADIUS_M, ge=0)
    simplify_tol_m: float = Field(DEFAULT_SIMPLIFY_TOL_M, ge=0)
    simplify_buffer_m: float = Field(DEFAULT_SIMPLIFY_BUFFER_M)
    include_zonal_stats: bool = True
    export_target: str = Field("zip")
    mode: str = Field(DEFAULT_MODE)
    ndvi_min: float | None = None
    ndvi_max: float | None = None

    @validator("months", each_item=True)
    def _validate_month(cls, value: str) -> str:
        month = str(value).strip()
        if not month:
            raise ValueError("Month entries cannot be empty")
        try:
            datetime.strptime(month, "%Y-%m")
        except ValueError as exc:
            raise ValueError(f"Invalid month format: {value}") from exc
        return month

    @validator("export_target")
    def _validate_target(cls, value: str) -> str:
        target = (value or "zip").strip().lower()
        if target not in {"zip", "local"}:
            raise ValueError("export_target must be 'zip' or 'local'")
        return target

    @model_validator(mode="after")
    def _ensure_month_inputs(self) -> "ProductionZonesRequest":
        months = self.months or []
        start = self.start_month
        end = self.end_month
        if not months and not (start and end):
            raise ValueError("Supply either months[] or start_month/end_month")
        return self


_zone_bundles: Dict[str, Dict[str, Any]] = {}
_zone_bundle_lock = threading.Lock()


def _purge_expired_zone_bundles(now: float | None = None) -> None:
    timestamp = now or time.time()
    expired: List[str] = []
    for token, payload in list(_zone_bundles.items()):
        created = payload.get("created", 0.0)
        if timestamp - created > ZONE_BUNDLE_TTL_SECONDS:
            expired.append(token)
    for token in expired:
        bundle = _zone_bundles.pop(token, None) or {}
        path = bundle.get("path")
        if isinstance(path, Path):
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass


def _register_zone_bundle(path: Path, filename: str) -> str:
    token = secrets.token_urlsafe(16)
    with _zone_bundle_lock:
        _purge_expired_zone_bundles()
        _zone_bundles[token] = {
            "path": path,
            "filename": filename,
            "created": time.time(),
        }
    return token


def _consume_zone_bundle(token: str) -> tuple[Path, str] | None:
    with _zone_bundle_lock:
        bundle = _zone_bundles.pop(token, None)
    if not bundle:
        return None
    path = bundle.get("path")
    filename = bundle.get("filename")
    if isinstance(path, Path) and isinstance(filename, str):
        return path, filename
    return None


def _create_zone_bundle(
    artifacts: ZoneArtifacts,
    prefix: str,
    metadata: Mapping[str, Any] | None,
    *,
    include_stats: bool,
) -> tuple[Path, str]:
    base_name = Path(prefix).name or sanitize_name(prefix or "zones")
    archive_filename = f"{base_name}_bundle.zip"
    temp_file = NamedTemporaryFile(delete=False, suffix=".zip")
    temp_path = Path(temp_file.name)
    temp_file.close()

    try:
        with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            mean_src = Path(artifacts.mean_ndvi_path)
            if not mean_src.exists():
                raise FileNotFoundError(f"Mean NDVI raster missing at {mean_src}")
            archive.write(mean_src, arcname=f"{base_name}_NDVI_mean.tif")

            raster_src = Path(artifacts.raster_path)
            if not raster_src.exists():
                raise FileNotFoundError(f"Classified raster missing at {raster_src}")
            archive.write(raster_src, arcname=f"{base_name}.tif")

            components = dict(artifacts.vector_components or {})
            if components and "geojson" not in components and artifacts.vector_path:
                components.setdefault("geojson", artifacts.vector_path)
            elif not components and artifacts.vector_path:
                components = {"geojson": artifacts.vector_path}

            for ext, src in components.items():
                dest_name = f"{base_name}.{str(ext).lower()}"
                if isinstance(src, (bytes, bytearray)):
                    archive.writestr(dest_name, bytes(src))
                    continue
                src_path = Path(str(src))
                if not src_path.exists():
                    continue
                archive.write(src_path, arcname=dest_name)

            if include_stats and artifacts.zonal_stats_path:
                stats_src_obj = artifacts.zonal_stats_path
                dest_name = f"{base_name}_zonal_stats.csv"
                if isinstance(stats_src_obj, (bytes, bytearray)):
                    archive.writestr(dest_name, bytes(stats_src_obj))
                else:
                    stats_src = Path(str(stats_src_obj))
                    if stats_src.exists():
                        archive.write(stats_src, arcname=dest_name)

            if metadata:
                try:
                    manifest = json.dumps(metadata, indent=2, sort_keys=True)
                    archive.writestr(f"{base_name}_metadata.json", manifest)
                except Exception:
                    pass
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise

    return temp_path, archive_filename

# NEW: zoning controls
DEFAULT_NDVI_MIN = 0.35
DEFAULT_NDVI_MAX = 0.73

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
    except Exception as exc:  # pragma: no cover
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
    except Exception:
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
    # Download vector features using EE's supported table formats.
    # Supported file_format values: 'geojson' | 'kml' | 'csv' | 'shp'
    fmt = (file_format or "geojson").strip().lower()
    filetype_map = {
        "geojson": "GeoJSON",
        "kml": "KML",
        "csv": "CSV",
        "shp": "SHP",
        "shapefile": "SHP",
    }
    filetype = filetype_map.get(fmt, "GeoJSON")

    if fmt == "geojson" and target.suffix.lower() != ".geojson":
        target = target.with_suffix(".geojson")
    if fmt == "kml" and target.suffix.lower() != ".kml":
        target = target.with_suffix(".kml")
    if fmt == "csv" and target.suffix.lower() != ".csv":
        target = target.with_suffix(".csv")
    if fmt in {"shp", "shapefile"} and target.suffix.lower() != ".zip":
        target = target.with_suffix(".zip")

    target.parent.mkdir(parents=True, exist_ok=True)

    attempted = []
    last_err = None
    for trial_filetype in ([filetype] + (["KML"] if filetype != "KML" else [])):
        try:
            attempted.append(trial_filetype)
            url = vectors.getDownloadURL(
                filetype=trial_filetype,
                selectors=["zone"],
                filename=target.stem
            )
            with urlopen(url) as response:
                headers = getattr(response, "headers", None)
                content_type = headers.get("Content-Type", "") if headers and hasattr(headers, "get") else ""
                if "zip" in content_type.lower() or trial_filetype == "SHP":
                    with NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
                        shutil.copyfileobj(response, tmp_zip)
                        tmp_zip_path = Path(tmp_zip.name)
                    try:
                        with ZipFile(tmp_zip_path, "r") as archive:
                            members = archive.namelist()
                            if not members:
                                raise ValueError("Vector download was empty")
                            pick = None
                            desired_ext = ".kml" if trial_filetype == "KML" else ".geojson"
                            for m in members:
                                if m.lower().endswith(desired_ext):
                                    pick = m
                                    break
                            if pick is None:
                                pick = members[0]
                            with archive.open(pick) as member:
                                target.write_bytes(member.read())
                    finally:
                        try:
                            tmp_zip_path.unlink(missing_ok=True)
                        except Exception:
                            pass
                else:
                    target.write_bytes(response.read())
            last_err = None
            break
        except Exception as e:
            last_err = e
            continue

    if last_err:
        raise ValueError(f"EE getDownloadURL failed after trying {attempted}: {last_err}")


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
    except TypeError:
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
    # Build mean NDVI for the selected period (warn on low variation, don't abort).
    def _iso(d):
        return d.isoformat() if isinstance(d, date) else str(d)

    monthly = gee.monthly_sentinel2_collection(
        aoi=geom,
        start=_iso(start_date),
        end=_iso(end_date),
        months=months,
        cloud_prob_max=cloud_prob_max,
    )

    try:
        ic_size = int(ee.Number(monthly.size()).getInfo() or 0)
    except Exception:
        ic_size = 0
    if ic_size == 0:
        raise ValueError(S2_COLLECTION_EMPTY_ERROR)

    def _ndvi(img: ee.Image) -> ee.Image:
        bnames = ee.List(img.bandNames())
        has_b4 = ee.Number(bnames.indexOf("B4")).gte(0)
        has_b8 = ee.Number(bnames.indexOf("B8")).gte(0)
        both = ee.Number(has_b4).multiply(ee.Number(has_b8)).eq(1)
        empty = ee.Image(0).updateMask(ee.Image(0)).rename("NDVI")
        ndvi = ee.Image(
            ee.Algorithms.If(
                both,
                img.toFloat().normalizedDifference(["B8", "B4"]).rename("NDVI"),
                empty,
            )
        )
        mask_1band = img.select("B8").mask().multiply(img.select("B4").mask()).gt(0)
        return ndvi.updateMask(mask_1band)

    ndvi_mean = monthly.map(_ndvi).mean().rename("NDVI_mean").toFloat().clip(geom)

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

    first_img = ee.Image(monthly.first())
    ndvi_native = ee.Algorithms.If(
        first_img,
        reproject_native_10m(ndvi_mean, first_img, ref_band="B8", scale=10),
        ndvi_mean,
    )

    return ee.Image(ndvi_native).rename("NDVI_mean").toFloat().clip(geom)


def _classify_smooth_and_polygonize(
    ndvi_mean_native: ee.Image,
    geom: ee.Geometry,
    *,
    n_zones: int = 3,
    mmu_ha: float = 1.0,
    smooth_radius_px: int = 1,
    mode: str = DEFAULT_MODE,
    ndvi_min: float | None = None,
    ndvi_max: float | None = None,
):
    ndvi = ndvi_mean_native.rename("NDVI_mean").clip(geom)

    lo = DEFAULT_NDVI_MIN if ndvi_min is None else float(ndvi_min)
    hi = DEFAULT_NDVI_MAX if ndvi_max is None else float(ndvi_max)
    if hi <= lo:
        hi = lo + 1e-6

    try:
        flat_prop = ndvi.get("ndvi_low_variation")
        flat_flag = str(flat_prop.getInfo()).lower()
    except Exception:
        flat_flag = "false"

    if mode == "quantile" or (mode == "auto" and flat_flag != "true"):
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
                thresholds_py.append(lo + (hi - lo) * (p / 100.0))
        method_used = "quantile"
    else:
        step = (hi - lo) / n_zones
        thresholds_py = [lo + i * step for i in range(n_zones + 1)]
        method_used = "linear"

    def classify(img: ee.Image, edges: List[float]) -> ee.Image:
        zones = ee.Image.constant(0)
        for i in range(len(edges) - 1):
            lower = edges[i]
            upper = edges[i + 1]
            zones = zones.where(img.gte(lower).And(img.lt(upper)), i + 1)
        zones = zones.where(img.gte(edges[-2]).And(img.lte(edges[-1])), len(edges) - 1)
        return zones.rename("zone").updateMask(img.mask()).toInt8()

    cls_raw = classify(ndvi, thresholds_py)

    cls_smooth = ee.Image(
        ee.Algorithms.If(
            ee.Number(smooth_radius_px).gt(0),
            cls_raw.focalMode(radius=smooth_radius_px, units="pixels"),
            cls_raw,
        )
    ).toInt8()

    min_px = ee.Number(mmu_ha).multiply(100).round().max(1)

    def keep_big(c):
        mask = cls_smooth.eq(c)
        valid = mask.connectedPixelCount(eightConnected=True).gte(min_px)
        return cls_smooth.updateMask(mask.And(valid))

    cls_mmu = ee.ImageCollection(
        [keep_big(ee.Number(c)) for c in range(1, n_zones + 1)]
    ).mosaic().rename("zone").toInt8().clip(geom)

    vectors = cls_mmu.reduceToVectors(
        geometry=geom,
        scale=10,
        geometryType="polygon",
        labelProperty="zone",
        reducer=ee.Reducer.countEvery(),
        bestEffort=True,
        maxPixels=1e9,
    )

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
    mode: str = DEFAULT_MODE,
    ndvi_min: float | None = None,
    ndvi_max: float | None = None,
) -> tuple[ZoneArtifacts, Dict[str, object]]:
    try:
        ordered_months = _ordered_months(months)
        skipped_months: List[str] = []

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

        ndvi_mean_native = _build_mean_ndvi_for_zones(
            geometry,
            start_date,
            end_date,
            months=ordered_months,
            cloud_prob_max=cloud_prob_max,
        )

        if ndvi_mean_native is None:
            raise ValueError("NDVI computation failed — Earth Engine returned an empty or invalid image.")

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
            raise ValueError("Zone classification failed — no valid NDVI variation or geometry produced.")

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

        vectors_reprojected = ee.FeatureCollection(vectors)

        geojson_path = workdir / "zones.geojson"
        kml_path = workdir / "zones.kml"
        _download_vector_to_path(vectors_reprojected, geojson_path, file_format="geojson")
        _download_vector_to_path(vectors_reprojected, kml_path, file_format="kml")

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

        palette = list(ZONE_PALETTE[: max(1, min(n_classes, len(ZONE_PALETTE)))])
        metadata: Dict[str, Any] = {
            "used_months": used_months,
            "skipped_months": skipped_months,
            "stability_mask_applied": False,
            "palette": palette,
            "requested_zone_count": int(n_classes),
            "effective_zone_count": int(len(unique_classes) or n_classes),
            "zones": zonal_stats,
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
    except ValueError:
        raise
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
        mode=mode,
        ndvi_min=ndvi_min,
        ndvi_max=ndvi_max,
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
    except Exception:
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


@router.post("/zones/production")
def create_production_zones(payload: ProductionZonesRequest) -> Dict[str, Any]:
    months = list(payload.months)
    if not months and payload.start_month and payload.end_month:
        try:
            start_dt = datetime.strptime(payload.start_month, "%Y-%m")
            end_dt = datetime.strptime(payload.end_month, "%Y-%m")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        start_date = date(start_dt.year, start_dt.month, 1)
        end_last_day = calendar.monthrange(end_dt.year, end_dt.month)[1]
        end_date = date(end_dt.year, end_dt.month, end_last_day)
        months = _months_from_dates(start_date, end_date)

    try:
        result = export_selected_period_zones(
            payload.aoi_geojson,
            payload.aoi_name,
            months,
            cloud_prob_max=payload.cloud_prob_max,
            n_classes=payload.n_classes,
            cv_mask_threshold=payload.cv_mask_threshold,
            min_mapping_unit_ha=payload.mmu_ha,
            smooth_radius_m=payload.smooth_radius_m,
            open_radius_m=payload.open_radius_m,
            close_radius_m=payload.close_radius_m,
            simplify_tolerance_m=int(round(payload.simplify_tol_m)),
            simplify_buffer_m=payload.simplify_buffer_m,
            export_target=payload.export_target,
            include_zonal_stats=payload.include_zonal_stats,
            include_stats=payload.include_zonal_stats,
            apply_stability_mask=payload.apply_stability_mask,
            mode=payload.mode,
            ndvi_min=payload.ndvi_min,
            ndvi_max=payload.ndvi_max,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Production zone export failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Zone export failed: {exc}") from exc

    artifacts = result.pop("artifacts", None)
    result.pop("working_dir", None)

    paths = result.get("paths") or {}
    if isinstance(paths, dict):
        cleaned_paths: Dict[str, Any] = {}
        for key, value in paths.items():
            if isinstance(value, str):
                cleaned_paths[key] = Path(value).name
            elif isinstance(value, Mapping):
                cleaned_paths[key] = {
                    ext: Path(str(name)).name if isinstance(name, str) else name
                    for ext, name in value.items()
                }
            else:
                cleaned_paths[key] = value
        result["paths"] = cleaned_paths

    metadata = result.get("metadata")
    if isinstance(metadata, dict):
        metadata.pop("downloaded_mean_ndvi", None)
        metadata.pop("mean_ndvi_export_task", None)
        metadata.pop("classified_export_task", None)

    download_token: str | None = None
    download_filename: str | None = None
    if (
        payload.export_target == "zip"
        and isinstance(artifacts, ZoneArtifacts)
    ):
        try:
            bundle_path, bundle_name = _create_zone_bundle(
                artifacts,
                result.get("prefix") or payload.aoi_name,
                metadata if isinstance(metadata, Mapping) else None,
                include_stats=payload.include_zonal_stats,
            )
            download_token = _register_zone_bundle(bundle_path, bundle_name)
            download_filename = bundle_name
        except Exception as exc:
            logger.exception("Failed to prepare zone bundle for download: %s", exc)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to prepare zone bundle: {exc}",
            ) from exc
        finally:
            if artifacts.working_dir:
                try:
                    shutil.rmtree(artifacts.working_dir, ignore_errors=True)
                except Exception:
                    pass
    elif isinstance(artifacts, ZoneArtifacts) and artifacts.working_dir:
        try:
            shutil.rmtree(artifacts.working_dir, ignore_errors=True)
        except Exception:
            pass

    response: Dict[str, Any] = {"ok": True, **result}
    if download_token:
        response["download_token"] = download_token
        response["download_filename"] = download_filename

    return response


@router.get("/zones/production/download/{token}")
def download_production_zone_bundle(token: str):
    bundle = _consume_zone_bundle(token)
    if not bundle:
        raise HTTPException(status_code=404, detail="Download expired or invalid")
    path, filename = bundle
    if not path.exists():
        raise HTTPException(status_code=410, detail="Bundle no longer available")

    def _stream() -> Any:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                yield chunk

    response = StreamingResponse(_stream(), media_type="application/zip")
    response.headers["Content-Disposition"] = f"attachment; filename=\"{filename}\""
    response.background = BackgroundTask(lambda: path.unlink(missing_ok=True))
    return response
