"""Production zone workflow built on Sentinel-2 monthly composites."""

from __future__ import annotations

import calendar
import csv
import io
import json
import logging
import math
import os
import re
import shutil
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import ee
import numpy as np
import rasterio
import shapefile
from pyproj import CRS
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import mapping, shape
from sklearn.cluster import KMeans

from app import gee
from app.exports import sanitize_name
from . import ee_utils
from .ee_utils import (
    cat_one as _cat_one,
    ensure_list as _ensure_list,
    ensure_number as _ensure_number,
    remove_nulls as _remove_nulls,
)
from .zones_core import (
    coverage_ratio,
    normalize_ndvi_band,
    stability_mask_from_cv,
    stats_stack,
)
from app.services.image_stats import temporal_stats
from app.utils.diag import Guard, PipelineError
from app.utils.geometry import area_ha
from app.utils.logging_colors import install_color_handler
from app.utils.sanitization import sanitize_for_json

logger = logging.getLogger(__name__)
install_color_handler(logger)
DEFAULT_CLOUD_PROB_MAX = 40
DEFAULT_N_CLASSES = 5
DEFAULT_CV_THRESHOLD = 0.5
DEFAULT_MIN_MAPPING_UNIT_HA = 1.5
DEFAULT_SMOOTH_RADIUS_M = 30
DEFAULT_OPEN_RADIUS_M = 10
DEFAULT_CLOSE_RADIUS_M = 10
DEFAULT_SIMPLIFY_TOL_M = 5
DEFAULT_SIMPLIFY_BUFFER_M = 3
DEFAULT_METHOD = "ndvi_kmeans"
DEFAULT_SAMPLE_SIZE = 8000
DEFAULT_SCALE = 10
# Holes below this threshold (in hectares) are removed during vector cleanup.
MIN_HOLE_AREA_HA = 0.1
# IMPORTANT: processing uses the native S2 projection (meters).
# Exports use a metric CRS so scale=10 means 10 meters.
DEFAULT_EXPORT_CRS = "EPSG:32756"
DEFAULT_CRS = DEFAULT_EXPORT_CRS


def _allow_init_failure() -> bool:
    flag = os.getenv("GEE_ALLOW_INIT_FAILURE", "")
    return flag.strip().lower() in {"1", "true", "yes"}


def ensure_list(value):
    ee_utils.ee = ee
    return _ensure_list(value)


def safe_ee_list(value):
    ee_utils.ee = ee
    return _ensure_list(value)


def remove_nulls(lst):
    ee_utils.ee = ee
    return _remove_nulls(lst)


def ensure_number(value):
    ee_utils.ee = ee
    return _ensure_number(value)


def cat_one(lst, value):
    ee_utils.ee = ee
    return _cat_one(lst, value)


def _to_ee_geometry(geojson: dict) -> ee.Geometry:
    """
    Accepts GeoJSON Geometry, Feature, or FeatureCollection and returns ee.Geometry.
    Raises ValueError on invalid input.
    """
    try:
        t = (geojson or {}).get("type", "")
        if t == "Feature":
            return ee.Feature(geojson).geometry()
        if t == "FeatureCollection":
            fc = ee.FeatureCollection(geojson)
            # union all geometries to one AOI (or use .geometry() for bbox)
            return fc.geometry()
        # assume raw Geometry
        return ee.Geometry(geojson)
    except Exception as e:
        raise ValueError(f"Invalid AOI GeoJSON: {e}")


def _parse_bool_env(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    trimmed = value.strip().lower()
    if trimmed in {"", "none"}:
        return default
    if trimmed in {"0", "false", "no", "off"}:
        return False
    if trimmed in {"1", "true", "yes", "on"}:
        return True
    return default


APPLY_STABILITY = _parse_bool_env(os.getenv("APPLY_STABILITY"), True)


def set_apply_stability(enabled: bool | None) -> None:
    """Temporarily override the APPLY_STABILITY flag.

    Passing ``None`` re-evaluates the environment variable.
    """
    global APPLY_STABILITY
    if enabled is None:
        APPLY_STABILITY = _parse_bool_env(os.getenv("APPLY_STABILITY"), True)
    else:
        APPLY_STABILITY = bool(enabled)


ZONE_PALETTE: tuple[str, ...] = (
    "#112f1d",
    "#1b4d2a",
    "#2c6a39",
    "#3f8749",
    "#58a35d",
    "#80bf7d",
    "#b6dcb1",
)

STABILITY_THRESHOLD_SEQUENCE = [0.5, 1.0, 1.5, 2.0]
MIN_STABILITY_SURVIVAL_RATIO = 0.0

NDVI_PERCENTILE_MIN = 0.0
NDVI_PERCENTILE_MAX = 0.6

NDVI_MASK_EMPTY_ERROR = (
    "No NDVI pixels were available for the selected period. Try expanding the date range "
    "or relaxing the cloud filtering threshold."
)

STABILITY_MASK_EMPTY_ERROR = (
    "All pixels were masked out by the stability threshold when computing NDVI percentiles. "
    "Try lowering the coefficient of variation threshold, expanding the selected months, "
    "or switching the zone method."
)


@dataclass(frozen=True)
class ZoneArtifacts:
    """Container for locally generated zone artefacts stored on disk."""

    raster_path: str
    mean_ndvi_path: str
    vector_path: str
    vector_components: dict[str, str]
    zonal_stats_path: str | None = None
    working_dir: str | None = None
    extra_files: dict[str, bytes] | None = None


@dataclass(frozen=True)
class ImageExportResult:
    """Result of staging an image locally while queueing an EE export task."""

    path: Path
    task: ee.batch.Task | None = None


def _ensure_working_directory(path: os.PathLike[str] | str | None) -> Path:
    if path:
        workdir = Path(path)
        workdir.mkdir(parents=True, exist_ok=True)
        return workdir
    return Path(tempfile.mkdtemp(prefix="zones_"))


def _as_lists(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_as_lists(item) for item in value]
    if isinstance(value, list):
        return [_as_lists(item) for item in value]
    return value


def _geometry_region(geometry: ee.Geometry) -> list[list[list[float]]]:
    info = geometry.getInfo()
    if not info:
        raise ValueError("Unable to resolve geometry information for download")

    geom_type = info.get("type")
    if geom_type == "Polygon":
        coordinates = info.get("coordinates")
        if coordinates:
            return _as_lists(coordinates)
        raise ValueError("Geometry information missing coordinates")

    if geom_type == "MultiPolygon":
        dissolved = geometry.dissolve()
        try:
            dissolved_info = dissolved.getInfo()
        except Exception:  # pragma: no cover - defensive guard for EE errors
            dissolved_info = None

        candidate_info = dissolved_info or info
        if not candidate_info or "coordinates" not in candidate_info:
            raise ValueError("Geometry information missing coordinates")

        if candidate_info.get("type") == "Polygon":
            coordinates = candidate_info.get("coordinates")
            if coordinates:
                return _as_lists(coordinates)

        merged_shape = shape(candidate_info)
        if merged_shape.is_empty:
            raise ValueError("Unable to resolve geometry information for download")
        if merged_shape.geom_type != "Polygon":
            merged_shape = merged_shape.convex_hull
        if merged_shape.geom_type != "Polygon":
            raise ValueError("Geometry information could not be merged into a polygon")
        mapped = mapping(merged_shape)
        return _as_lists(mapped.get("coordinates", []))

    coordinates = info.get("coordinates")
    if coordinates is not None:
        return _as_lists(coordinates)
    raise ValueError("Geometry information missing coordinates")


def _is_zip_payload(content_type: str | None, payload: bytes) -> bool:
    if content_type and "zip" in content_type.lower():
        return True
    return (
        payload.startswith(b"PK\x03\x04")
        or payload.startswith(b"PK\x05\x06")
        or payload.startswith(b"PK\x07\x08")
    )


def _extract_geotiff_from_zip(payload: bytes, target: Path) -> None:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [
            info
            for info in archive.infolist()
            if not info.is_dir() and info.filename.lower().endswith((".tif", ".tiff"))
        ]
        if not members:
            raise ValueError("Zip archive did not contain a GeoTIFF file")
        member = members[0]
        with archive.open(member) as source, target.open("wb") as output:
            shutil.copyfileobj(source, output)


def _download_image_to_path(
    image: ee.Image, geometry: ee.Geometry, target: Path
) -> ImageExportResult:
    region_coords = _geometry_region(geometry)
    ee_region = ee.Geometry.Polygon(region_coords)
    sanitized_name = sanitize_name(target.stem or "export")
    description = f"zones_{sanitized_name}"[:100]
    folder = os.getenv("GEE_DRIVE_FOLDER", "Sentinel2_Zones")

    task: ee.batch.Task | None = None
    try:
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=description,
            folder=folder,
            fileNamePrefix=sanitized_name,
            region=ee_region,
            scale=DEFAULT_SCALE,
            crs=DEFAULT_EXPORT_CRS,
            fileFormat="GeoTIFF",
            maxPixels=gee.MAX_PIXELS,
            filePerBand=False,
        )
        task.start()
    except Exception:  # pragma: no cover - diagnostic logging
        logger.exception("Failed to start Drive export for %s", sanitized_name)
        task = None

    params = {
        "scale": DEFAULT_SCALE,
        "crs": DEFAULT_EXPORT_CRS,
        "region": region_coords,
        "filePerBand": False,
        "format": "GeoTIFF",
    }
    url = image.getDownloadURL(params)
    with urlopen(url) as response:
        payload = response.read()
        headers = getattr(response, "headers", None)
        content_type = ""
        if headers is not None and hasattr(headers, "get"):
            content_type = headers.get("Content-Type", "")
        else:  # pragma: no cover - fallback for alternative urllib implementations
            getheader = getattr(response, "getheader", None)
            if callable(getheader):
                content_type = getheader("Content-Type", "")

    target.parent.mkdir(parents=True, exist_ok=True)
    if _is_zip_payload(content_type, payload):
        _extract_geotiff_from_zip(payload, target)
    else:
        with target.open("wb") as output:
            output.write(payload)
    return ImageExportResult(path=target, task=task)


def _majority_filter(data: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return data

    size = radius * 2 + 1

    def _mode(window: np.ndarray) -> int:
        values = window.astype(np.int32)
        values = values[values > 0]
        if values.size == 0:
            return 0
        counts = np.bincount(values)
        return int(np.argmax(counts))

    filtered = ndimage.generic_filter(data, _mode, size=size, mode="nearest")
    zero_mask = data == 0
    filtered[zero_mask] = 0
    return filtered.astype(data.dtype)


def _apply_morphological_operation(
    data: np.ndarray, radius: int, *, operation: str
) -> np.ndarray:
    if radius <= 0:
        return data

    size = radius * 2 + 1
    footprint = np.ones((size, size), dtype=bool)
    valid_mask = data > 0
    if not np.any(valid_mask):
        return data

    working = data.copy()
    working[~valid_mask] = 0

    if operation == "opening":
        filtered = ndimage.grey_opening(working, footprint=footprint, mode="nearest")
    elif operation == "closing":
        filtered = ndimage.grey_closing(working, footprint=footprint, mode="nearest")
    else:  # pragma: no cover - defensive branch
        raise ValueError(f"Unsupported morphological operation: {operation}")

    filtered[~valid_mask] = 0
    return filtered.astype(data.dtype)


def _remove_small_regions(data: np.ndarray, min_pixels: int) -> np.ndarray:
    if min_pixels <= 1:
        return data

    structure = ndimage.generate_binary_structure(data.ndim, 2)
    result = data.copy()
    for value in np.unique(result):
        if value <= 0:
            continue
        mask = result == value
        labeled, num = ndimage.label(mask, structure=structure)
        if num == 0:
            continue
        sizes = ndimage.sum(mask, labeled, index=range(1, num + 1))
        small_labels = [idx + 1 for idx, size in enumerate(sizes) if size < min_pixels]
        if not small_labels:
            continue
        remove_mask = np.isin(labeled, small_labels)
        result[remove_mask] = 0
    return result


def _assemble_zone_artifacts(
    *,
    classified: np.ndarray,
    ndvi: np.ma.MaskedArray,
    transform,
    crs,
    working_dir: Path,
    include_stats: bool,
    raster_path: Path,
    mean_ndvi_path: Path,
    smoothing_requested: Mapping[str, float],
    applied_operations: Mapping[str, bool],
    executed_operations: Mapping[str, bool],
    fallback_applied: bool,
    fallback_removed: Sequence[str],
    min_mapping_unit_ha: float,
    requested_zone_count: int,
    effective_zone_count: int,
    classification_method: str,
    thresholds: Sequence[float],
    kmeans_fallback_applied: bool,
    kmeans_cluster_centers: Sequence[float],
) -> tuple[ZoneArtifacts, dict[str, object]]:
    pixel_size_x = abs(transform.a)
    pixel_size_y = abs(transform.e)
    pixel_area = pixel_size_x * pixel_size_y

    classified_array = np.asarray(classified)
    if classified_array.dtype != np.uint8:
        classified_array = classified_array.astype(np.uint8)

    unique_zones = np.unique(classified_array[classified_array > 0])

    records: list[dict[str, object]] = []
    zonal_stats: list[dict[str, object]] = []
    for geom, value in shapes(
        classified_array, mask=classified_array > 0, transform=transform
    ):
        zone_id = int(value)
        if zone_id <= 0:
            continue
        geom_shape = shape(geom)
        area_m2 = geom_shape.area
        area_ha = area_m2 / 10_000.0
        records.append({"zone": zone_id, "geometry": geom_shape, "area_ha": area_ha})

    vector_path = working_dir / "zones"
    vector_path.mkdir(parents=True, exist_ok=True)
    shp_base = vector_path / "zones"
    with shapefile.Writer(str(shp_base)) as writer:
        writer.autoBalance = 1
        writer.field("zone", "N", decimal=0)
        writer.field("area_ha", "F", decimal=4)
        for record in records:
            writer.record(int(record["zone"]), float(record["area_ha"]))
            writer.shape(mapping(record["geometry"]))

    shp_path = shp_base.with_suffix(".shp")
    cpg_path = shp_base.with_suffix(".cpg")
    cpg_path.write_text("UTF-8")
    if crs:
        prj_path = shp_base.with_suffix(".prj")
        try:
            prj_path.write_text(CRS.from_user_input(crs).to_wkt())
        except Exception:  # pragma: no cover - pyproj failures
            prj_path.write_text("")

    vector_components: dict[str, str] = {}
    for ext in ["shp", "dbf", "shx", "prj", "cpg"]:
        component = shp_base.with_suffix(f".{ext}")
        if component.exists():
            vector_components[ext] = str(component)

    ndvi_data = ndvi.filled(np.nan)
    for zone_id in unique_zones:
        mask = classified_array == zone_id
        zone_values = ndvi_data[mask]
        zone_values = zone_values[~np.isnan(zone_values)]
        if zone_values.size == 0:
            continue
        area_ha = float(mask.sum() * pixel_area / 10_000.0)
        zonal_stats.append(
            {
                "zone": int(zone_id),
                "area_ha": area_ha,
                "mean_ndvi": float(np.mean(zone_values)),
                "min_ndvi": float(np.min(zone_values)),
                "max_ndvi": float(np.max(zone_values)),
                "pixel_count": int(mask.sum()),
            }
        )

    stats_path: Path | None = None
    if include_stats and zonal_stats:
        stats_path = working_dir / "zones_zonal_stats.csv"
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

    smoothing_requested = {
        key: float(max(value, 0)) for key, value in dict(smoothing_requested).items()
    }
    stage_key_map = {
        "smooth": "smooth_radius_m",
        "open": "open_radius_m",
        "close": "close_radius_m",
        "min_mapping_unit": "min_mapping_unit_ha",
    }
    smoothing_applied = {
        metadata_key: (
            smoothing_requested.get(metadata_key, 0.0)
            if applied_operations.get(stage_name)
            else 0.0
        )
        for stage_name, metadata_key in stage_key_map.items()
    }
    smoothing_executed = {
        stage_key_map[name]: bool(executed_operations.get(name))
        for name in stage_key_map
    }

    skipped_steps = [
        stage_key_map[name] for name in fallback_removed if name in stage_key_map
    ]
    skipped_steps = sorted(set(skipped_steps))

    final_zone_count = int(unique_zones.size)
    palette: list[str] = list(
        ZONE_PALETTE[: max(1, min(final_zone_count, len(ZONE_PALETTE)))]
    )

    metadata: dict[str, object] = {
        "percentile_thresholds": [float(value) for value in thresholds],
        "palette": palette,
        "zones": zonal_stats,
        "min_mapping_unit_applied": bool(applied_operations.get("min_mapping_unit")),
        "mmu_applied": bool(applied_operations.get("min_mapping_unit")),
        "smoothing_parameters": {
            "requested": smoothing_requested,
            "applied": smoothing_applied,
            "executed": smoothing_executed,
            "fallback_applied": bool(fallback_applied),
            "skipped_steps": skipped_steps,
            "rolled_back_steps": skipped_steps,
        },
        "requested_zone_count": int(requested_zone_count),
        "effective_zone_count": int(effective_zone_count),
        "final_zone_count": final_zone_count,
    }

    metadata["kmeans_fallback_applied"] = bool(kmeans_fallback_applied)
    metadata["kmeans_cluster_centers"] = [
        float(value) for value in kmeans_cluster_centers
    ]
    metadata["classification_method"] = classification_method

    artifacts = ZoneArtifacts(
        raster_path=str(raster_path),
        mean_ndvi_path=str(mean_ndvi_path),
        vector_path=str(shp_path),
        vector_components=vector_components,
        zonal_stats_path=str(stats_path) if stats_path is not None else None,
        working_dir=str(working_dir),
    )

    return artifacts, metadata


def _ensure_float_image(image: ee.Image) -> ee.Image:
    to_float = getattr(image, "toFloat", None)
    if callable(to_float):
        try:
            return to_float()
        except Exception:  # pragma: no cover - defensive fallback
            logger.exception("Failed to convert image to float; using original")
    return image


def _classify_local_zones(
    ndvi_raster: Path,
    *,
    working_dir: Path,
    n_classes: int,
    min_mapping_unit_ha: float,
    smooth_radius_m: float,
    open_radius_m: float,
    close_radius_m: float,
    include_stats: bool,
    forced_thresholds: Sequence[float] | None = None,
) -> tuple[ZoneArtifacts, dict[str, object]]:
    with rasterio.open(ndvi_raster) as src:
        ndvi = src.read(1, masked=True)
        transform = src.transform
        crs = src.crs
        profile = src.profile

    if ndvi.mask.all():
        raise ValueError(NDVI_MASK_EMPTY_ERROR)

    ndvi_data = ndvi.filled(np.nan)
    combined_mask = np.ma.getmaskarray(ndvi) | np.isnan(ndvi_data)
    valid_mask = ~combined_mask
    if combined_mask.all():
        raise ValueError(NDVI_MASK_EMPTY_ERROR)

    valid_values = ndvi_data[~combined_mask]
    if valid_values.size == 0:
        raise ValueError(NDVI_MASK_EMPTY_ERROR)

    unique_values = np.unique(valid_values)
    if unique_values.size <= 1:
        raise ValueError(
            "Unable to derive distinct NDVI thresholds for zone classification; all pixels share the same value."
        )

    effective_n_classes = n_classes

    vmin = float(np.min(valid_values))
    vmax = float(np.max(valid_values))

    percentiles = np.linspace(0, 100, effective_n_classes + 1)[1:-1]
    thresholds = np.array([], dtype=float)

    if forced_thresholds is not None:
        forced = np.asarray([float(value) for value in forced_thresholds], dtype=float)
        forced = forced[np.isfinite(forced)]
        forced = np.unique(forced)
        forced.sort()
        if forced.size >= max(effective_n_classes - 1, 0):
            thresholds = forced[: max(effective_n_classes - 1, 0)]
        if thresholds.size and not np.all(np.diff(thresholds) > 0):
            thresholds = np.array([], dtype=float)

    if not thresholds.size:
        raw_thresholds = (
            np.nanpercentile(valid_values, percentiles)
            if percentiles.size
            else np.array([])
        )

        thresholds = raw_thresholds
        if thresholds.size:
            thresholds = np.asarray(thresholds, dtype=float)
            thresholds = thresholds[~np.isnan(thresholds)]
            thresholds = np.unique(thresholds)
            if thresholds.size < max(effective_n_classes - 1, 0):
                thresholds = np.linspace(vmin, vmax, effective_n_classes + 1)[1:-1]
            if thresholds.size != max(effective_n_classes - 1, 0) or not np.all(
                np.diff(thresholds) > 0
            ):
                thresholds = np.array([], dtype=float)

    comparison_data = ndvi_data[..., None]
    if thresholds.size:
        classified = np.sum(comparison_data > thresholds, axis=-1, dtype=np.int16) + 1
    else:
        classified = np.ones(ndvi_data.shape, dtype=np.int16)
    classified[combined_mask] = 0

    unique_zones = np.unique(classified[classified > 0])
    if (
        unique_zones.size < effective_n_classes
        and effective_n_classes > 1
        and percentiles.size
        and unique_values.size >= effective_n_classes
    ):
        unique_sorted, counts = np.unique(valid_values, return_counts=True)
        if unique_sorted.size >= effective_n_classes:
            num_thresholds = effective_n_classes - 1
            if num_thresholds > 0:
                boundary_props = np.cumsum(counts)[:-1] / float(counts.sum())
                if boundary_props.size >= num_thresholds and boundary_props.size > 0:
                    B = boundary_props.size
                    selected_indices: list[int] = []
                    prev_idx = -1
                    for j, target in enumerate(percentiles / 100.0):
                        min_idx = max(prev_idx + 1, j)
                        max_idx = B - (num_thresholds - j - 1) - 1
                        max_idx = min(max_idx, B - 1)
                        if max_idx < min_idx:
                            min_idx = max_idx
                        found_idx = int(
                            np.searchsorted(boundary_props, target, side="left")
                        )
                        if found_idx >= B:
                            found_idx = B - 1
                        idx = min(max(found_idx, min_idx), max_idx)
                        selected_indices.append(idx)
                        prev_idx = idx
                    fallback_thresholds = np.array(
                        [
                            (float(unique_sorted[idx]) + float(unique_sorted[idx + 1]))
                            / 2.0
                            for idx in selected_indices
                        ],
                        dtype=float,
                    )
                    fallback_comparison = comparison_data
                    fallback_classified = (
                        np.sum(
                            fallback_comparison > fallback_thresholds,
                            axis=-1,
                            dtype=np.int16,
                        )
                        + 1
                    )
                    fallback_classified[combined_mask] = 0
                    fallback_unique = np.unique(
                        fallback_classified[fallback_classified > 0]
                    )
                    if fallback_unique.size == effective_n_classes:
                        thresholds = fallback_thresholds
                        classified = fallback_classified
                        unique_zones = fallback_unique

    pixel_size_x = abs(transform.a)
    pixel_size_y = abs(transform.e)
    pixel_area = pixel_size_x * pixel_size_y
    radius_scale = max(pixel_size_x, pixel_size_y)

    smooth_radius_px = int(round(max(smooth_radius_m, 0) / radius_scale))
    open_radius_px = int(round(max(open_radius_m, 0) / radius_scale))
    close_radius_px = int(round(max(close_radius_m, 0) / radius_scale))

    stage_results: list[tuple[str, np.ndarray]] = [("initial", classified.copy())]
    applied_operations: dict[str, bool] = {
        "smooth": False,
        "open": False,
        "close": False,
        "min_mapping_unit": False,
    }
    executed_operations: dict[str, bool] = {
        "smooth": False,
        "open": False,
        "close": False,
        "min_mapping_unit": False,
    }

    if smooth_radius_px > 0:
        classified = _majority_filter(classified, smooth_radius_px)
        stage_results.append(("smooth", classified.copy()))
        applied_operations["smooth"] = True
        executed_operations["smooth"] = True
    if open_radius_px > 0:
        classified = _apply_morphological_operation(
            classified, open_radius_px, operation="opening"
        )
        stage_results.append(("open", classified.copy()))
        applied_operations["open"] = True
        executed_operations["open"] = True
    if close_radius_px > 0:
        classified = _apply_morphological_operation(
            classified, close_radius_px, operation="closing"
        )
        stage_results.append(("close", classified.copy()))
        applied_operations["close"] = True
        executed_operations["close"] = True

    min_pixels = int(round((min_mapping_unit_ha * 10_000) / pixel_area))
    if min_pixels > 1:
        classified = _remove_small_regions(classified, min_pixels)
        stage_results.append(("min_mapping_unit", classified.copy()))
        applied_operations["min_mapping_unit"] = True
        executed_operations["min_mapping_unit"] = True

    fallback_removed: list[str] = []
    final_index = len(stage_results) - 1
    unique_zones = np.unique(
        stage_results[final_index][1][stage_results[final_index][1] > 0]
    )
    while unique_zones.size < effective_n_classes and final_index > 0:
        stage_name = stage_results[final_index][0]
        if stage_name in applied_operations and applied_operations[stage_name]:
            applied_operations[stage_name] = False
            fallback_removed.append(stage_name)
        final_index -= 1
        unique_zones = np.unique(
            stage_results[final_index][1][stage_results[final_index][1] > 0]
        )

    fallback_removed.reverse()
    fallback_applied = bool(fallback_removed)

    classified = stage_results[final_index][1].copy()
    unique_zones = np.unique(classified[classified > 0])
    kmeans_cluster_centers: list[float] | None = None
    kmeans_fallback_applied = False

    if unique_zones.size < effective_n_classes:
        if effective_n_classes <= 0:
            raise ValueError(
                "Zone classification produced fewer zones than requested even after relaxing smoothing "
                f"operations ({unique_zones.size} < {effective_n_classes}). Final thresholds: {thresholds.tolist()}"
            )

        kmeans = KMeans(n_clusters=effective_n_classes, n_init=10, random_state=0)
        kmeans.fit(valid_values.reshape(-1, 1))
        centers = kmeans.cluster_centers_.reshape(-1)
        center_order = np.argsort(centers)
        zone_lookup = np.zeros(center_order.size, dtype=np.uint8)
        for zone_index, cluster_index in enumerate(center_order, start=1):
            zone_lookup[cluster_index] = np.uint8(zone_index)

        fallback_classified = np.zeros(ndvi_data.shape, dtype=np.uint8)
        fallback_classified[~combined_mask] = zone_lookup[kmeans.labels_]
        classified = fallback_classified
        unique_zones = np.unique(classified[classified > 0])
        thresholds = np.array([], dtype=float)
        kmeans_cluster_centers = [float(centers[idx]) for idx in center_order]
        kmeans_fallback_applied = True

        if unique_zones.size < effective_n_classes:
            if valid_values.size >= effective_n_classes:
                sorted_indices = np.argsort(valid_values, kind="mergesort")
                splits = np.array_split(sorted_indices, effective_n_classes)
                if all(split.size for split in splits):
                    fallback_flat = np.zeros(valid_values.size, dtype=np.uint8)
                    for class_index, split in enumerate(splits, start=1):
                        fallback_flat[split] = np.uint8(class_index)
                    ranked_classified = np.zeros(ndvi_data.shape, dtype=np.uint8)
                    ranked_classified[~combined_mask] = fallback_flat
                    classified = ranked_classified
                    unique_zones = np.unique(classified[classified > 0])

        if unique_zones.size < effective_n_classes:
            raise ValueError(
                "Zone classification produced fewer zones than requested even after the K-means fallback "
                f"({unique_zones.size} < {effective_n_classes})."
            )

    if classified.dtype != np.uint8:
        classified = classified.astype(np.uint8)

    if np.any(classified == 0):
        fill_mask = valid_mask & (classified == 0)
        if np.any(fill_mask):
            filled = _majority_filter(classified, 1)
            classified[fill_mask] = filled[fill_mask]
    unique_zones = np.unique(classified[classified > 0])

    def _hex_to_rgba(value: str) -> tuple[int, int, int, int]:
        value = value.lstrip("#")
        return (
            int(value[0:2], 16),
            int(value[2:4], 16),
            int(value[4:6], 16),
            255,
        )

    colormap: dict[int, tuple[int, int, int, int]] = {0: (0, 0, 0, 0)}
    if unique_zones.size:
        for zone_id in unique_zones:
            palette_index = min(len(ZONE_PALETTE) - 1, int(zone_id) - 1)
            colormap[int(zone_id)] = _hex_to_rgba(ZONE_PALETTE[palette_index])

    raster_profile = profile.copy()
    raster_profile.update(dtype=rasterio.uint8, count=1, compress="lzw", nodata=0)

    raster_path = working_dir / "zones_classified.tif"
    with rasterio.open(raster_path, "w", **raster_profile) as dst:
        dst.write(classified, 1)
        if hasattr(dst, "write_colormap"):
            dst.write_colormap(1, colormap)

    smoothing_requested = {
        "smooth_radius_m": float(max(smooth_radius_m, 0)),
        "open_radius_m": float(max(open_radius_m, 0)),
        "close_radius_m": float(max(close_radius_m, 0)),
        "min_mapping_unit_ha": float(max(min_mapping_unit_ha, 0)),
    }

    artifacts, metadata = _assemble_zone_artifacts(
        classified=classified,
        ndvi=ndvi,
        transform=transform,
        crs=crs,
        working_dir=working_dir,
        include_stats=include_stats,
        raster_path=raster_path,
        mean_ndvi_path=ndvi_raster,
        smoothing_requested=smoothing_requested,
        applied_operations=applied_operations,
        executed_operations=executed_operations,
        fallback_applied=fallback_applied,
        fallback_removed=fallback_removed,
        min_mapping_unit_ha=min_mapping_unit_ha,
        requested_zone_count=n_classes,
        effective_zone_count=effective_n_classes,
        classification_method="kmeans" if kmeans_fallback_applied else "percentiles",
        thresholds=thresholds.tolist() if thresholds.size else [],
        kmeans_fallback_applied=kmeans_fallback_applied,
        kmeans_cluster_centers=(
            kmeans_cluster_centers if kmeans_cluster_centers is not None else []
        ),
    )

    return artifacts, metadata


def _ordered_months(months: Sequence[str]) -> list[str]:
    unique: dict[str, datetime] = {}
    for raw in months:
        month_str = str(raw).strip()
        if month_str in unique:
            continue
        try:
            parsed = datetime.strptime(month_str, "%Y-%m")
        except ValueError as exc:
            raise ValueError(f"Invalid month format: {raw}") from exc
        unique[month_str] = parsed

    ordered = sorted(unique.items(), key=lambda item: item[1])
    return [month for month, _ in ordered]


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


def _months_from_dates(start_date: date, end_date: date) -> list[str]:
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")
    months: list[str] = []
    cursor = date(start_date.year, start_date.month, 1)
    end_cursor = date(end_date.year, end_date.month, 1)
    while cursor <= end_cursor:
        months.append(cursor.strftime("%Y-%m"))
        if cursor.month == 12:
            cursor = cursor.replace(year=cursor.year + 1, month=1)
        else:
            cursor = cursor.replace(month=cursor.month + 1)
    return months


def _month_bounds(months: Sequence[str]) -> tuple[str, str]:
    if not months:
        raise ValueError("At least one month must be supplied")
    ordered = _ordered_months(months)
    return ordered[0], ordered[-1]


def _export_prefix(aoi_name: str, months: Sequence[str]) -> str:
    start, end = _month_bounds(months)
    safe_name = sanitize_name(aoi_name or "aoi")
    return (
        f"zones/PROD_{start.replace('-', '')}_{end.replace('-', '')}_{safe_name}_zones"
    )


def export_prefix(aoi_name: str, months: Sequence[str]) -> str:
    """Public helper that returns the export prefix for a zone export run."""
    return _export_prefix(aoi_name, months)


def month_bounds(months: Sequence[str]) -> tuple[str, str]:
    """Return the first and last months used in an export request."""
    return _month_bounds(months)


def resolve_export_bucket(explicit: str | None = None) -> str:
    bucket = (
        explicit or os.getenv("GEE_GCS_BUCKET") or os.getenv("GCS_BUCKET") or ""
    ).strip()
    if not bucket:
        raise RuntimeError("GEE_GCS_BUCKET or GCS_BUCKET must be set for zone exports")
    return bucket


def _resolve_geometry(aoi: dict | ee.Geometry) -> ee.Geometry:
    try:
        geometry_type = ee.Geometry
        if isinstance(geometry_type, type) and isinstance(aoi, geometry_type):
            return aoi
    except TypeError:
        pass
    return gee.geometry_from_geojson(aoi)


def _attach_cloud_probability(
    collection: ee.ImageCollection, probability: ee.ImageCollection
) -> ee.ImageCollection:
    join = ee.Join.saveFirst("cloud_prob")
    matches = join.apply(
        primary=collection,
        secondary=probability,
        condition=ee.Filter.equals(leftField="system:index", rightField="system:index"),
    )

    def _add_probability(image: ee.Image) -> ee.Image:
        cloud_match = image.get("cloud_prob")
        probability_band = ee.Image(
            ee.Algorithms.If(
                ee.Algorithms.IsEqual(cloud_match, None),
                ee.Image.constant(0),
                ee.Image(cloud_match).select("probability"),
            )
        ).rename("cloud_probability")
        return image.addBands(probability_band)

    return ee.ImageCollection(matches).map(_add_probability)


def _mask_sentinel2_scene(image: ee.Image, cloud_prob_max: int) -> ee.Image:
    qa = image.select("QA60")
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    qa_mask = (
        qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    )

    prob_mask = image.select("cloud_probability").lte(cloud_prob_max)

    scl = image.select("SCL")
    shadow_mask = scl.neq(3).And(scl.neq(11))

    combined_mask = qa_mask.And(prob_mask).And(shadow_mask)
    scaled = image.updateMask(combined_mask).divide(10_000)
    selected = scaled.select(list(gee.S2_BANDS))
    return selected.copyProperties(image, ["system:time_start"])


# --- NDVI & SCL helpers (strict) --------------------------------------------
def apply_s2_cloud_mask(img: ee.Image) -> ee.Image:
    # Keep vegetation/bare/water; strict baseline
    scl = img.select("SCL")
    keep = scl.remap([4, 5, 6], [1, 1, 1], 0)
    return img.updateMask(keep)


def monthly_ndvi_composite(aoi, start, end) -> ee.Image:
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start, end)
        .map(apply_s2_cloud_mask)
        .map(compute_ndvi)
    )
    monthly = col.median()
    monthly = _ndvi_1band(monthly)
    return monthly.updateMask(monthly.mask())


def build_monthly_ndvi_collection(
    aoi,
    months: list[str],
    *,
    mask_mode: str,
    min_valid_ratio: float,
    cloud_prob_max: int,
) -> ee.ImageCollection:
    def _range(ym: str):
        y, m = ym.split("-")
        start = ee.Date.fromYMD(int(y), int(m), 1)
        end = start.advance(1, "month")
        return start, end

    images = []
    for ym in months:
        s, e = _range(ym)
        monthly_image = monthly_ndvi_adaptive(
            aoi,
            s,
            e,
            mask_mode=mask_mode,
            min_valid_ratio=min_valid_ratio,
            cloud_prob_max=cloud_prob_max,
        )
        monthly = _ndvi_1band(monthly_image).set(
            {"ym": ym, "mask_tier": monthly_image.get("mask_tier")}
        )
        images.append(monthly)

    monthly_ndvi = ee.ImageCollection(images)
    monthly_ndvi = monthly_ndvi.filter(
        ee.Filter.listContains("system:band_names", "NDVI")
    )
    return monthly_ndvi


def _native_reproject(img: ee.Image) -> ee.Image:
    """Reproject to native Sentinel-2 10 m projection (meters)."""
    # Use B8 (10 m) projection as the reference
    proj = img.select("B8").projection()
    return img.resample("bilinear").reproject(proj, None, DEFAULT_SCALE)


def _build_masked_s2_collection(
    *_args, **_kwargs
):  # pragma: no cover - compatibility shim
    raise NotImplementedError(
        "_build_masked_s2_collection is not implemented in this module"
    )


def _build_composite_collection(
    *_args, **_kwargs
):  # pragma: no cover - compatibility shim
    raise NotImplementedError(
        "_build_composite_collection is not implemented in this module"
    )


def _ndvi_collection(*_args, **_kwargs):  # pragma: no cover - compatibility shim
    raise NotImplementedError("_ndvi_collection is not implemented in this module")


def _ndvi_statistics(*_args, **_kwargs):  # pragma: no cover - compatibility shim
    raise NotImplementedError("_ndvi_statistics is not implemented in this module")


def _ndvi_percentile_thresholds(
    *_args, **_kwargs
):  # pragma: no cover - compatibility shim
    raise NotImplementedError(
        "_ndvi_percentile_thresholds is not implemented in this module"
    )


def _classify_percentiles(*_args, **_kwargs):  # pragma: no cover - compatibility shim
    raise NotImplementedError("_classify_percentiles is not implemented in this module")


def _vectorize_zones(*_args, **_kwargs):  # pragma: no cover - compatibility shim
    raise NotImplementedError("_vectorize_zones is not implemented in this module")


def _zonal_statistics(*_args, **_kwargs):  # pragma: no cover - compatibility shim
    raise NotImplementedError("_zonal_statistics is not implemented in this module")


def _build_composite_series(
    geometry: ee.Geometry,
    months: Sequence[str],
    start_date: date,
    end_date: date,
    cloud_prob_max: int,
    mask_mode: str,
    min_valid_ratio: float,
) -> tuple[list[tuple[str, ee.Image]], list[str], dict[str, object]]:
    composites: list[tuple[str, ee.Image]] = []
    skipped: list[str] = []
    metadata: dict[str, object] = {}
    ordered = _ordered_months(months)
    month_span = len(ordered)

    start_iso = start_date.isoformat()
    end_exclusive_iso = (end_date + timedelta(days=1)).isoformat()

    if month_span >= 3:
        metadata["composite_mode"] = "monthly"
        for month in ordered:
            collection, composite = gee.monthly_sentinel2_collection(
                geometry, month, cloud_prob_max
            )
            try:
                scene_count = int(ee.Number(collection.size()).getInfo() or 0)
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    f"Failed to evaluate Sentinel-2 collection for {month}: {exc}"
                )
            if scene_count == 0:
                skipped.append(month)
                continue

            try:
                month_dt = datetime.strptime(month, "%Y-%m")
            except ValueError as exc:
                raise ValueError(f"Invalid month string: {month}") from exc

            month_start = ee.Date.fromYMD(month_dt.year, month_dt.month, 1)
            month_end = month_start.advance(1, "month")

            ndvi_image = _ndvi_1band(
                monthly_ndvi_adaptive(
                    geometry,
                    month_start,
                    month_end,
                    mask_mode=mask_mode,
                    min_valid_ratio=min_valid_ratio,
                    cloud_prob_max=cloud_prob_max,
                )
            )

            try:
                valid_pixels = int(
                    ee.Number(
                        ndvi_image.mask()
                        .reduceRegion(
                            reducer=ee.Reducer.count(),
                            geometry=geometry,
                            scale=DEFAULT_SCALE,
                            bestEffort=True,
                            tileScale=4,
                            maxPixels=gee.MAX_PIXELS,
                        )
                        .get("NDVI")
                    ).getInfo()
                    or 0
                )
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    f"Failed to determine valid pixel count for {month}: {exc}"
                ) from exc

            if valid_pixels == 0:
                skipped.append(month)
                continue

            reproj = _native_reproject(composite)
            reproj = reproj.updateMask(ndvi_image.mask())
            composite_with_ndvi = reproj.addBands(ndvi_image, overwrite=True)
            composite_with_ndvi = composite_with_ndvi.set(
                {"mask_tier": ndvi_image.get("mask_tier")}
            )
            composites.append((month, composite_with_ndvi.clip(geometry)))
    else:
        metadata["composite_mode"] = "scene"
        metadata["start_date"] = start_iso
        metadata["end_date"] = end_exclusive_iso
        metadata["end_date_inclusive"] = end_date.isoformat()

        base_collection = (
            ee.ImageCollection(gee.S2_SR_COLLECTION)
            .filterBounds(geometry)
            .filterDate(start_iso, end_exclusive_iso)
        )
        probability = (
            ee.ImageCollection(gee.S2_CLOUD_PROB_COLLECTION)
            .filterBounds(geometry)
            .filterDate(start_iso, end_exclusive_iso)
        )

        with_prob = _attach_cloud_probability(base_collection, probability)
        masked = with_prob.map(lambda img: _mask_sentinel2_scene(img, cloud_prob_max))

        try:
            scene_count = int(ee.Number(masked.size()).getInfo() or 0)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                f"Failed to evaluate Sentinel-2 scene collection for {start_iso} to {end_exclusive_iso}: {exc}"
            ) from exc

        metadata["scene_count"] = scene_count
        if scene_count == 0:
            return [], ordered, metadata

        image_list = masked.toList(scene_count)
        for idx in range(scene_count):
            image = ee.Image(image_list.get(idx))
            # ⬇⬇ FIX: native projection (meters)
            reproj = _native_reproject(image)
            label = f"scene_{idx + 1:02d}"
            composites.append((label, reproj.clip(geometry)))

    return composites, skipped, metadata


def compute_ndvi(img: ee.Image) -> ee.Image:
    b8 = img.select("B8").toFloat()
    b4 = img.select("B4").toFloat()
    ndvi = b8.subtract(b4).divide(b8.add(b4).add(1e-6)).rename("NDVI")
    return ndvi.updateMask(b8.mask().And(b4.mask()))


# --- Helpers -----------------------------------------------------------------
def _ndvi_from(img: ee.Image) -> ee.Image:
    b8 = img.select("B8").toFloat()
    b4 = img.select("B4").toFloat()
    ndvi = b8.subtract(b4).divide(b8.add(b4).add(1e-6)).rename("NDVI")
    return ndvi.updateMask(b8.mask().And(b4.mask()))


def _ndvi_1band(img: ee.Image) -> ee.Image:
    # Accepts images that might have 'NDVI' or 'NDVI_mean'; returns single band named 'NDVI'
    bnames = img.bandNames()
    has_ndvi = bnames.contains("NDVI")
    has_ndvi_mean = bnames.contains("NDVI_mean")
    img2 = ee.Image(
        ee.Algorithms.If(
            has_ndvi,
            img.select(["NDVI"]).rename(["NDVI"]),
            ee.Algorithms.If(
                has_ndvi_mean,
                img.select(["NDVI_mean"]).rename(["NDVI"]),
                img,  # fallback; will be caught by guards if wrong
            ),
        )
    )
    # Ensure single-band
    return ee.Image(img2).select(["NDVI"]).rename(["NDVI"])


def _ndvi_std(img: ee.Image) -> ee.Image:
    # Returns a band named 'NDVI_stdDev' from either 'NDVI_stdDev' or 'NDVI'
    bnames = img.bandNames()
    has_std = bnames.contains("NDVI_stdDev")
    has_ndvi = bnames.contains("NDVI")
    return ee.Image(
        ee.Algorithms.If(
            has_std,
            img.select(["NDVI_stdDev"]).rename(["NDVI_stdDev"]),
            ee.Algorithms.If(
                has_ndvi,
                img.select(["NDVI"]).rename(["NDVI_stdDev"]),
                img,
            ),
        )
    )


def _scl_mask(img: ee.Image, keep_codes: Sequence[int]) -> ee.Image:
    return (
        img.select("SCL").remap(keep_codes, [1] * len(keep_codes), 0).rename("SCL_keep")
    )


def _cover_ratio(
    img: ee.Image, region: ee.Geometry, scale: int = 10, tile_scale: int = 4
) -> ee.Number:
    valid_sum = img.mask().reduceRegion(
        ee.Reducer.sum(),
        region,
        scale,
        maxPixels=1e9,
        bestEffort=True,
        tileScale=tile_scale,
    )
    total_px = region.area(1).divide(ee.Number(scale).pow(2))
    return ee.Number(valid_sum.values().reduce(ee.Reducer.sum())).divide(total_px)


# --- Adaptive monthly composite ---------------------------------------------
def monthly_ndvi_adaptive(
    aoi,
    start,
    end,
    *,
    mask_mode: str,
    min_valid_ratio: float,
    cloud_prob_max: int,
) -> ee.Image:
    geometry = aoi
    if not isinstance(geometry, ee.Geometry):
        try:
            geometry = _to_ee_geometry(aoi)
        except Exception:
            geometry = ee.Geometry(aoi)

    try:
        region = (
            ee.FeatureCollection([ee.Feature(geometry)]).geometry().buffer(5).bounds(1)
        )
    except Exception:
        region = geometry

    base = (
        ee.ImageCollection(gee.S2_SR_COLLECTION)
        .filterBounds(region)
        .filterDate(start, end)
    )

    tiers: list[dict[str, object]] = []
    tiers.append(
        {
            "name": "strict",
            "apply": lambda ic: ic.map(
                lambda im: _ndvi_from(im).updateMask(_scl_mask(im, [4, 5, 6]))
            ),
        }
    )
    tiers.append(
        {
            "name": "relaxed",
            "apply": lambda ic: ic.map(
                lambda im: _ndvi_from(im).updateMask(_scl_mask(im, [3, 4, 5, 6, 7]))
            ),
        }
    )
    tiers.append({"name": "minimal", "apply": lambda ic: ic.map(_ndvi_from)})

    tiers_by_name = {tier["name"]: tier for tier in tiers}
    tier_order = (
        ["strict"]
        if mask_mode == "strict"
        else ["relaxed"] if mask_mode == "relaxed" else ["strict", "relaxed", "minimal"]
    )

    candidates: list[ee.Image] = []
    for idx, name in enumerate(tier_order):
        tier_def = tiers_by_name.get(name)
        if tier_def is None:
            continue
        ndvi_ic = ee.ImageCollection(tier_def["apply"](base))
        comp = ndvi_ic.median()
        comp = _ndvi_1band(comp)
        ratio = _cover_ratio(comp, region)
        comp = comp.set(
            {
                "mask_tier": name,
                "valid_ratio": ratio,
                "tier_index": idx,
            }
        )
        candidates.append(comp)

    minimal_def = tiers_by_name["minimal"]
    minimal_ic = ee.ImageCollection(minimal_def["apply"](base))
    minimal_comp = minimal_ic.median()
    minimal_comp = _ndvi_1band(minimal_comp)
    minimal_ratio = _cover_ratio(minimal_comp, region)
    minimal_comp = minimal_comp.set(
        {
            "mask_tier": "minimal",
            "valid_ratio": minimal_ratio,
            "tier_index": len(tier_order),
        }
    )

    if not candidates:
        candidates.append(minimal_comp)

    cand_ic = ee.ImageCollection(candidates)
    acceptable = (
        cand_ic.filter(ee.Filter.gte("valid_ratio", min_valid_ratio))
        .sort("tier_index")
        .first()
    )

    fallback_minimal = minimal_comp.set(
        {
            "mask_tier": "fallback_minimal",
            "valid_ratio": minimal_comp.get("valid_ratio"),
        }
    )
    comp_img = ee.Image(
        ee.Algorithms.If(
            acceptable,
            acceptable,
            fallback_minimal,
        )
    )
    return comp_img.set("mask_tier", comp_img.get("mask_tier"))


# Backwards compatibility for callers still referencing the private helper
_compute_ndvi = compute_ndvi


def _compute_ndre(image: ee.Image) -> ee.Image:
    return image.normalizedDifference(["B8", "B5"]).rename("NDRE")


def _compute_ndmi(image: ee.Image) -> ee.Image:
    return image.normalizedDifference(["B8", "B11"]).rename("NDMI")


def _compute_bsi(image: ee.Image) -> ee.Image:
    return image.expression(
        "((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue))",
        {
            "swir1": image.select("B11"),
            "red": image.select("B4"),
            "nir": image.select("B8"),
            "blue": image.select("B2"),
        },
    ).rename("BSI")


def _as_image_collection(
    images: Sequence[ee.Image] | ee.ImageCollection,
) -> ee.ImageCollection:
    if hasattr(images, "map") and hasattr(images, "size"):
        return ee.ImageCollection(images)
    return ee.ImageCollection(list(images))


def _ndvi_temporal_stats(
    images: Sequence[ee.Image] | ee.ImageCollection,
) -> Mapping[str, ee.Image]:
    stats = temporal_stats(
        images,
        band_name="NDVI",
        rename_prefix="NDVI",
    )
    return {
        "mean": stats["mean"],
        "median": stats["median"],
        "std": stats["std"],
        "cv": stats["cv"],
    }


def _stability_mask(
    cv_image: ee.Image,
    geometry: ee.Geometry,
    thresholds: Sequence[float],
    min_survival_ratio: float,
    scale: int,
) -> ee.Image:
    total = ee.Number(
        cv_image.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=geometry,
            scale=scale,
            bestEffort=True,
            tileScale=4,
            maxPixels=gee.MAX_PIXELS,
        )
        .values()
        .get(0)
    )

    logger.debug(
        "Stability thresholds (type=%s): %s",
        type(thresholds).__name__,
        thresholds,
    )
    threshold_list = safe_ee_list([float(t) for t in thresholds])
    min_ratio = ee.Number(min_survival_ratio)

    def _mask_for_threshold(value):
        t = ee.Number(value)
        raw_mask = cv_image.lte(t)
        masked = ee.Image(raw_mask).selfMask()
        surviving = ee.Number(
            masked.reduceRegion(
                reducer=ee.Reducer.count(),
                geometry=geometry,
                scale=scale,
                bestEffort=True,
                tileScale=4,
                maxPixels=gee.MAX_PIXELS,
            )
            .values()
            .get(0)
        )
        ratio = surviving.divide(total.max(1))
        return ee.Image(
            ee.Algorithms.If(
                ratio.gte(min_ratio),
                masked,
                ee.Image(0).selfMask(),
            )
        )

    masks = threshold_list.map(_mask_for_threshold)
    combined = ee.ImageCollection.fromImages(masks).max()

    combined_masked = ee.Image(combined).selfMask()
    combined_count = ee.Number(
        combined_masked.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=geometry,
            scale=scale,
            bestEffort=True,
            tileScale=4,
            maxPixels=gee.MAX_PIXELS,
        )
        .values()
        .get(0)
    )

    pass_through = ee.Image(1)
    return ee.Image(
        ee.Algorithms.If(combined_count.lte(0), pass_through, combined_masked)
    ).selfMask()


def _pixel_count(
    image: ee.Image,
    geometry: ee.Geometry,
    *,
    context: str,
    scale: int = DEFAULT_SCALE,
) -> int:
    try:
        reduced = image.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=geometry,
            scale=scale,
            bestEffort=True,
            tileScale=4,
            maxPixels=gee.MAX_PIXELS,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to evaluate {context}: {exc}") from exc

    try:
        reduced_info = reduced.getInfo() if hasattr(reduced, "getInfo") else reduced
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to evaluate {context}: {exc}") from exc

    if isinstance(reduced_info, dict):
        values = list(reduced_info.values())
        value = values[0] if values else 0
    else:
        value = reduced_info

    try:
        numeric = float(value or 0)
    except Exception:  # pragma: no cover
        numeric = 0.0

    if not math.isfinite(numeric):
        numeric = 0.0

    return int(max(numeric, 0))


def _percentile_thresholds(
    reducer_dict: Mapping[str, float], percentiles: Sequence[float], label: str
) -> list[float]:
    """
    Build the list of percentile thresholds for NDVI zoning.

    Accepts reducer outputs where percentile keys are either:
      - 'cut_01', 'cut_02', ... (bare keys)
      - '<label>_cut_01', '<label>_cut_02', ... (band-prefixed keys from EE)
      - 'p20', 'p40', ... (Earth Engine defaults)
      - '<label>_p20', ... (Earth Engine defaults with band prefix)

    Args:
        reducer_dict: dictionary returned by EE reduceRegion with percentiles
        percentiles: ordered list of requested percentile values (0..100)
        label: band name prefix (e.g. 'ndvi_mean')

    Returns:
        A list of thresholds in ascending order
    """
    if not percentiles:
        raise ValueError("percentiles must be a non-empty sequence")

    label_prefix = f"{label}_" if label else ""
    cut_lookup: dict[int, float] = {}
    percentile_lookup: list[tuple[float, float]] = []

    cut_pattern = re.compile(r"^cut_(\d+)$")
    pct_pattern = re.compile(r"^p(\d+(?:_\d+)*)$")

    for raw_key, raw_value in reducer_dict.items():
        key = str(raw_key)
        if label_prefix and key.startswith(label_prefix):
            key = key[len(label_prefix) :]

        cut_match = cut_pattern.match(key)
        if cut_match:
            try:
                ordinal = int(cut_match.group(1))
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                cut_lookup[ordinal] = value
            continue

        pct_match = pct_pattern.match(key)
        if pct_match:
            suffix = pct_match.group(1).replace("_", ".")
            try:
                pct_value = float(suffix)
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(pct_value) and math.isfinite(value):
                percentile_lookup.append((pct_value, value))

    percentile_lookup.sort(key=lambda item: item[0])

    thresholds: list[float] = []
    remaining_pct = percentile_lookup
    for ordinal, pct in enumerate(percentiles, start=1):
        try:
            target = float(pct)
        except (TypeError, ValueError):
            raise ValueError("percentiles must be numeric") from None

        if ordinal in cut_lookup:
            thresholds.append(cut_lookup[ordinal])
            continue

        match_index = None
        for idx, (pct_value, value) in enumerate(remaining_pct):
            if math.isclose(pct_value, target, rel_tol=1e-6, abs_tol=0.5):
                match_index = idx
                break

        if match_index is None:
            raise ValueError(STABILITY_MASK_EMPTY_ERROR)

        _, value = remaining_pct.pop(match_index)
        thresholds.append(value)

    return thresholds


def _classify_by_percentiles(
    image: ee.Image, geometry: ee.Geometry, n_classes: int
) -> tuple[ee.Image, list[float]]:
    """
    Classify an NDVI image into percentile-based zones.

    Steps:
    - ReduceRegion computes percentile thresholds (n_classes - 1 cuts).
    - _percentile_thresholds interprets both bare 'cut_XX' and band-prefixed keys.
    - Classify pixels by counting how many thresholds their value exceeds.
    """

    # Ensure image has a known band name
    band_name_obj = ee.String(image.bandNames().get(0))
    image = image.rename(band_name_obj)
    if hasattr(band_name_obj, "getInfo"):
        band_label = band_name_obj.getInfo()
    else:
        band_label = str(band_name_obj)

    # Percentile cuts to request
    step = 100 / n_classes
    percentile_sequence = [step * i for i in range(1, n_classes)]
    logger.debug(
        "Percentile sequence (type=%s): %s",
        type(percentile_sequence).__name__,
        percentile_sequence,
    )
    pct_breaks = safe_ee_list(percentile_sequence)
    output_names = [f"cut_{i:02d}" for i in range(1, n_classes)]

    # Compute percentiles for this band
    reducer_dict = image.reduceRegion(
        reducer=ee.Reducer.percentile(pct_breaks, outputNames=output_names),
        geometry=geometry,
        scale=DEFAULT_SCALE,
        bestEffort=True,
        tileScale=4,
        maxPixels=gee.MAX_PIXELS,
    )

    # Convert reducer result into Python dict (safe)
    reducer_info = reducer_dict.getInfo() or {}

    # Extract thresholds with robust handling of bare/prefixed keys
    thresholds: list[float]
    try:
        thresholds = _percentile_thresholds(
            reducer_info, percentile_sequence, band_label
        )
    except ValueError as exc:
        raise ValueError(STABILITY_MASK_EMPTY_ERROR) from exc

    adjusted_thresholds: list[float] = []
    previous = -math.inf
    for raw_value in thresholds:
        try:
            numeric = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Percentile thresholds must be numeric") from exc

        if not math.isfinite(numeric):
            raise ValueError("Percentile thresholds must be finite values")

        if numeric <= previous:
            nudged = math.nextafter(previous, math.inf)
            if not math.isfinite(nudged) or nudged <= previous:
                raise ValueError(
                    "Unable to derive strictly increasing percentile thresholds"
                )
            numeric = nudged

        adjusted_thresholds.append(numeric)
        previous = numeric

    thresholds = adjusted_thresholds
    logger.debug(
        "Adjusted percentile thresholds (type=%s): %s",
        type(thresholds).__name__,
        thresholds,
    )

    # Now classify pixels relative to thresholds
    zero = image.multiply(0)

    def _accumulate(current, threshold):
        current_img = ee.Image(current)
        t = ee.Number(threshold)
        gt_band = image.gt(t)
        return current_img.add(gt_band)

    summed = safe_ee_list(thresholds).iterate(_accumulate, zero)
    classified = ee.Image(summed).add(1).toInt()

    return classified.rename("zone"), thresholds


def _connected_component_area(classified: ee.Image, n_classes: int) -> ee.Image:
    pixel_area = ee.Image.pixelArea()
    band_names = classified.bandNames()
    area_image = ee.Image.constant(0).rename(band_names)
    for class_id in range(1, n_classes + 1):
        mask = classified.eq(class_id)
        counts = mask.connectedPixelCount(maxSize=1_000_000, eightConnected=True)
        area = counts.multiply(pixel_area).rename(band_names)
        area_image = area_image.where(mask, area)
    return area_image.rename("component_area")


def _apply_cleanup(
    classified: ee.Image,
    geometry: ee.Geometry,
    *,
    n_classes: int,
    smooth_radius_m: float,
    open_radius_m: float,
    close_radius_m: float,
    min_mapping_unit_ha: float,
) -> ee.Image:
    smooth_radius = max(float(smooth_radius_m), 0.0)
    open_radius = max(float(open_radius_m), 0.0)
    close_radius = max(float(close_radius_m), 0.0)

    smoothed = classified
    if smooth_radius > 0:
        smoothed = classified.focal_mode(
            radius=smooth_radius, units="meters", iterations=1
        )

    opened = smoothed
    if open_radius > 0:
        opened = smoothed.focal_min(
            radius=open_radius, units="meters", iterations=1
        ).focal_max(radius=open_radius, units="meters", iterations=1)

    closed = opened
    if close_radius > 0:
        closed = opened.focal_max(
            radius=close_radius, units="meters", iterations=1
        ).focal_min(radius=close_radius, units="meters", iterations=1)

    component_area = _connected_component_area(closed, n_classes)
    min_area_m2 = max(min_mapping_unit_ha, 0) * 10_000
    majority_large = closed
    if smooth_radius > 0:
        majority_large = closed.focal_mode(
            radius=smooth_radius, units="meters", iterations=1
        )
    small_mask = component_area.lt(min_area_m2)
    if hasattr(small_mask, "rename") and hasattr(classified, "bandNames"):
        small_mask = small_mask.rename(classified.bandNames())
    cleaned = closed.where(small_mask, majority_large)

    mask = cleaned.mask()
    closed_mask = mask
    if close_radius > 0:
        closed_mask = mask.focal_max(
            radius=close_radius, units="meters", iterations=1
        ).focal_min(radius=close_radius, units="meters", iterations=1)
    filler = cleaned
    if smooth_radius > 0:
        filler = cleaned.focal_mode(radius=smooth_radius, units="meters", iterations=1)
    cleaned = cleaned.where(mask.Not(), filler)
    cleaned = cleaned.updateMask(closed_mask)
    cleaned = cleaned.clip(geometry)
    return cleaned


@dataclass
class CleanupResult:
    image: ee.Image
    applied_operations: dict[str, bool]
    executed_operations: dict[str, bool]
    fallback_applied: bool
    fallback_removed: list[str]


def _unique_zone_count(image: ee.Image, geometry: ee.Geometry) -> int:
    try:
        result = image.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=geometry,
            scale=DEFAULT_SCALE,
            bestEffort=True,
            tileScale=4,
            maxPixels=gee.MAX_PIXELS,
        )
        info = result.getInfo()
    except Exception:
        return 0

    if not isinstance(info, Mapping):
        return 0

    histogram: Mapping[object, object] | None = None
    for value in info.values():
        if isinstance(value, Mapping):
            histogram = value
            break

    if not histogram:
        return 0

    def _as_positive_number(raw: object) -> float | None:
        if isinstance(raw, (int, float)):
            return float(raw)
        try:
            return float(str(raw))
        except (TypeError, ValueError):
            return None

    count = 0
    for key in histogram.keys():
        numeric = _as_positive_number(key)
        if numeric is not None and numeric > 0:
            count += 1

    return count


def _apply_cleanup_with_fallback_tracking(
    classified: ee.Image,
    geometry: ee.Geometry,
    *,
    n_classes: int,
    smooth_radius_m: float,
    open_radius_m: float,
    close_radius_m: float,
    min_mapping_unit_ha: float,
) -> CleanupResult:
    mmu_value = max(float(min_mapping_unit_ha), 0.0)

    applied_operations: dict[str, bool] = {
        "smooth": bool(smooth_radius_m > 0),
        "open": bool(open_radius_m > 0),
        "close": bool(close_radius_m > 0),
        "min_mapping_unit": bool(mmu_value > 0),
    }
    executed_operations = dict(applied_operations)

    stage_names: list[str] = []
    stage_images: list[ee.Image] = []
    stage_counts: list[int] = []

    def _append_stage(name: str, image: ee.Image) -> None:
        stage_names.append(name)
        stage_images.append(image)
        stage_counts.append(_unique_zone_count(image, geometry))

    base_image = _apply_cleanup(
        classified,
        geometry,
        n_classes=n_classes,
        smooth_radius_m=0.0,
        open_radius_m=0.0,
        close_radius_m=0.0,
        min_mapping_unit_ha=0.0,
    )
    _append_stage("initial", base_image)

    if applied_operations["smooth"]:
        smooth_image = _apply_cleanup(
            classified,
            geometry,
            n_classes=n_classes,
            smooth_radius_m=smooth_radius_m,
            open_radius_m=0.0,
            close_radius_m=0.0,
            min_mapping_unit_ha=0.0,
        )
        _append_stage("smooth", smooth_image)

    if applied_operations["open"]:
        open_image = _apply_cleanup(
            classified,
            geometry,
            n_classes=n_classes,
            smooth_radius_m=smooth_radius_m,
            open_radius_m=open_radius_m,
            close_radius_m=0.0,
            min_mapping_unit_ha=0.0,
        )
        _append_stage("open", open_image)

    if applied_operations["close"]:
        close_image = _apply_cleanup(
            classified,
            geometry,
            n_classes=n_classes,
            smooth_radius_m=smooth_radius_m,
            open_radius_m=open_radius_m,
            close_radius_m=close_radius_m,
            min_mapping_unit_ha=0.0,
        )
        _append_stage("close", close_image)

    if applied_operations["min_mapping_unit"]:
        mmu_image = _apply_cleanup(
            classified,
            geometry,
            n_classes=n_classes,
            smooth_radius_m=smooth_radius_m,
            open_radius_m=open_radius_m,
            close_radius_m=close_radius_m,
            min_mapping_unit_ha=mmu_value,
        )
        _append_stage("min_mapping_unit", mmu_image)

    fallback_removed: list[str] = []
    final_index = len(stage_images) - 1
    while final_index > 0 and stage_counts[final_index] < n_classes:
        stage_name = stage_names[final_index]
        if stage_name in applied_operations:
            applied_operations[stage_name] = False
            fallback_removed.append(stage_name)
        final_index -= 1

    fallback_removed.reverse()
    fallback_applied = bool(fallback_removed)

    final_image = stage_images[final_index]

    return CleanupResult(
        image=final_image,
        applied_operations=dict(applied_operations),
        executed_operations=dict(executed_operations),
        fallback_applied=fallback_applied,
        fallback_removed=list(fallback_removed),
    )


def _clean_zones(
    classified: ee.Image,
    geometry: ee.Geometry,
    *,
    smooth_radius_m: float,
    open_radius_m: float,
    close_radius_m: float,
    min_mapping_unit_ha: float,
    n_classes: int | None = None,
) -> ee.Image:
    image = ee.Image(classified)
    try:
        return _apply_cleanup(
            image,
            geometry,
            n_classes=DEFAULT_N_CLASSES if n_classes is None else n_classes,
            smooth_radius_m=smooth_radius_m,
            open_radius_m=open_radius_m,
            close_radius_m=close_radius_m,
            min_mapping_unit_ha=min_mapping_unit_ha,
        )
    except AttributeError:
        return image.clip(geometry)


def _simplify_vectors(
    vectors: ee.FeatureCollection, tolerance_m: float, buffer_m: float
) -> ee.FeatureCollection:
    def _simplify(feature: ee.Feature) -> ee.Feature:
        geom = feature.geometry()
        if tolerance_m > 0:
            geom = geom.simplify(maxError=tolerance_m)
        if buffer_m != 0:
            geom = geom.buffer(buffer_m)
        zone_value = ee.Number(feature.get("zone")).toInt()
        area_m2 = geom.area(maxError=1)
        area_ha_val = area_m2.divide(10_000)
        return feature.setGeometry(geom).set(
            {
                "zone": zone_value,
                "zone_id": zone_value,
                "area_m2": area_m2,
                "area_ha": area_ha_val,
            }
        )

    return ee.FeatureCollection(vectors.map(_simplify))


def _dissolve_vectors(
    vectors: ee.FeatureCollection, *, min_hole_area_ha: float
) -> ee.FeatureCollection:
    histogram = ee.Dictionary(vectors.aggregate_histogram("zone"))
    zone_ids = histogram.keys()
    hole_area_m2 = ee.Number(min_hole_area_ha).multiply(10_000)

    def _dissolve(zone_value: ee.ComputedObject) -> ee.Feature:
        zone_num = ee.Number(zone_value)
        zone_vectors = vectors.filter(ee.Filter.eq("zone", zone_num))
        base_feature = ee.Feature(zone_vectors.first())
        dissolved_geom = zone_vectors.geometry(maxError=1).dissolve()
        cleaned_geom = ee.Geometry(
            ee.Algorithms.If(
                hole_area_m2.gt(0),
                ee.Geometry(dissolved_geom).removeInteriorHoles(hole_area_m2),
                ee.Geometry(dissolved_geom),
            )
        )
        return base_feature.setGeometry(cleaned_geom).set(
            {"zone": zone_num, "zone_id": zone_num}
        )

    return ee.FeatureCollection(zone_ids.map(_dissolve))


def _prepare_vectors(
    zone_image: ee.Image,
    geometry: ee.Geometry,
    *,
    tolerance_m: float,
    buffer_m: float,
) -> ee.FeatureCollection:
    vectors = zone_image.reduceToVectors(
        geometry=geometry,
        scale=20,
        maxPixels=gee.MAX_PIXELS,
        geometryType="polygon",
        eightConnected=True,
        labelProperty="zone",
        reducer=ee.Reducer.first(),
    )

    def _set_zone(feature: ee.Feature) -> ee.Feature:
        zone_value = ee.Number(feature.get("zone")).toInt()
        return feature.set({"zone": zone_value, "zone_id": zone_value})

    vectors = vectors.map(_set_zone)
    vectors = _dissolve_vectors(vectors, min_hole_area_ha=MIN_HOLE_AREA_HA)

    return _simplify_vectors(vectors, tolerance_m, buffer_m)


def _collect_stats_images(
    ndvi_stats: Mapping[str, ee.Image],
    extra_means: Mapping[str, ee.Image] | None = None,
) -> dict[str, ee.Image]:
    stats: dict[str, ee.Image] = {
        "NDVI_mean": ndvi_stats["mean"],
        "NDVI_median": ndvi_stats["median"],
        "NDVI_stdDev": ndvi_stats["std"],
        "NDVI_cv": ndvi_stats["cv"],
    }
    if extra_means:
        for key, image in extra_means.items():
            stats[key] = image
    return stats


def _add_zonal_stats(
    feature: ee.Feature, stats_images: Mapping[str, ee.Image]
) -> ee.Feature:
    geometry = feature.geometry()
    area_ha_val = geometry.area(maxError=1).divide(10_000)
    ordered = [stats_images[name].rename(name) for name in sorted(stats_images)]
    stats_image = ee.Image.cat(ordered)
    stats_dict = ee.Dictionary(
        stats_image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=DEFAULT_SCALE,
            bestEffort=True,
            tileScale=4,
            maxPixels=gee.MAX_PIXELS,
        )
    )

    keys = stats_dict.keys()

    def _sanitize_value(key) -> ee.Dictionary:
        key_str = ee.String(key)
        value = stats_dict.get(key_str)

        number_ctor = getattr(ee, "Number", None)
        if isinstance(value, (int, float)) and number_ctor is not None:
            return ee.Dictionary({}).set(key_str, number_ctor(value))

        if value is None:
            return ee.Dictionary({}).set(key_str, None)

        image_cls = getattr(ee, "Image", None)
        try:
            if image_cls is not None and isinstance(value, image_cls):
                return ee.Dictionary({})
        except TypeError:
            # ``ee.Image`` may not be a Python type in some fakes.
            pass

        try:
            if number_ctor is not None and isinstance(value, number_ctor):
                return ee.Dictionary({}).set(key_str, number_ctor(value))
        except TypeError:
            # ``ee.Number`` may not be a Python type in fake implementations.
            pass

        value_type = ee.String(ee.Algorithms.ObjectType(value))
        is_number = value_type.compareTo("Number").eq(0)
        is_image = value_type.compareTo("Image").eq(0)

        return ee.Dictionary(
            ee.Algorithms.If(
                is_image,
                ee.Dictionary({}),
                ee.Algorithms.If(
                    is_number,
                    ee.Dictionary({}).set(key_str, value),
                    ee.Dictionary({}).set(key_str, None),
                ),
            )
        )

    def _merge(entry: ee.ComputedObject, acc: ee.ComputedObject) -> ee.Dictionary:
        return ee.Dictionary(acc).combine(ee.Dictionary(entry))

    sanitized_stats = ee.Dictionary(
        keys.map(_sanitize_value).iterate(_merge, ee.Dictionary({}))
    )
    zone_value = ee.Number(feature.get("zone")).toInt()
    return feature.set(sanitized_stats).set(
        {"area_ha": area_ha_val, "zone": zone_value, "zone_id": zone_value}
    )


def _build_percentile_zones(
    *,
    ndvi_stats: Mapping[str, ee.Image],
    geometry: ee.Geometry,
    n_classes: int,
    smooth_radius_m: float,
    open_radius_m: float,
    close_radius_m: float,
    min_mapping_unit_ha: float,
) -> tuple[ee.Image, list[float]]:
    # Cap NDVI for percentile breaks only (0..0.6)
    pct_source = ndvi_stats["mean"].clamp(NDVI_PERCENTILE_MIN, NDVI_PERCENTILE_MAX)

    # Robust thresholds: compute on mean mask (not stability) to avoid empty stats
    thresholds_image = pct_source.updateMask(ndvi_stats["mean"].mask())
    ranked_for_thresh, thresholds = _classify_by_percentiles(
        thresholds_image, geometry, n_classes
    )
    # Now classify the full pct_source, then apply stability mask afterwards
    ranked_full, _ = _classify_by_percentiles(pct_source, geometry, n_classes)
    ranked = ranked_full.updateMask(ndvi_stats["stability"])

    try:
        percentile_thresholds = [float(value) for value in thresholds]
    except (TypeError, ValueError) as exc:  # pragma: no cover
        raise RuntimeError(
            f"Failed to evaluate NDVI percentile thresholds: {exc}"
        ) from exc

    cleaned = _apply_cleanup(
        ranked,
        geometry,
        n_classes=n_classes,
        smooth_radius_m=smooth_radius_m,
        open_radius_m=open_radius_m,
        close_radius_m=close_radius_m,
        min_mapping_unit_ha=min_mapping_unit_ha,
    )
    return cleaned.rename("zone"), percentile_thresholds


# --- Robust quantile breaks with self-healing --------------------------------
def robust_quantile_breaks(
    ndvi_img: ee.Image,
    aoi,
    n_classes: int,
    scale: int = 10,
    tile_scale: int = 4,
) -> ee.List:
    """
    Returns strictly increasing K-1 breaks. Tries, in order:
    1) native percentiles on tiny jittered floats
    2) histogram-based quantiles (fixedHistogram)
    3) fallback: return ee.List([]) to trigger k-means
    """
    logger.debug(
        "robust_quantile_breaks: starting with n_classes=%d, scale=%d", n_classes, scale
    )

    region = ee.FeatureCollection(aoi).geometry().buffer(5).bounds(1)
    # ensure a stable band name for reducer outputs
    base_band = ndvi_img.select([0]).rename("NDVI")
    # ensure float + tiny jitter to break exact ties without changing classing
    base = base_band.toFloat().add(ee.Image.random().multiply(1e-6))

    # 1) native percentiles
    perc = [int(round(100 * i / n_classes)) for i in range(1, n_classes)]
    logger.debug(
        "Robust quantile percentiles (type=%s): %s",
        type(perc).__name__,
        perc,
    )
    try1 = base.reduceRegion(
        reducer=ee.Reducer.percentile(perc, None),  # key order NDVI_pXX
        geometry=region,
        scale=scale,
        maxPixels=1e9,
        bestEffort=True,
        tileScale=tile_scale,
    )

    try:
        vals1 = safe_ee_list(perc).map(lambda p: try1.get(f"NDVI_p{p}"))
        logger.debug("robust_quantile_breaks: percentile method succeeded")
    except Exception as e:
        logger.error(
            "robust_quantile_breaks: FAILED at percentile mapping | error=%s: %s",
            type(e).__name__,
            str(e),
            exc_info=True,
        )
        raise

    # de-dup & enforce increasing using ee logic
    def _uniq_sort(values):
        try:
            l2 = safe_ee_list(values).sort()
            # remove nulls
            l2 = remove_nulls(l2)
            size = ee.Number(l2.size())
            logger.debug("_uniq_sort: sorted and removed nulls")
        except Exception as e:
            logger.error(
                "_uniq_sort: FAILED during sort/remove_nulls | error=%s: %s",
                type(e).__name__,
                str(e),
                exc_info=True,
            )
            raise

        # squash equal neighbors by adding epsilon steps
        def _dedup(idx, prev):
            try:
                idx = ensure_number(idx)
                prev_type = ee.String(ee.Algorithms.ObjectType(prev))
                prev_is_list = prev_type.compareTo("List").eq(0)
                acc = safe_ee_list(ee.Algorithms.If(prev_is_list, prev, ee.List([])))
                v = ensure_number(l2.get(idx))
                return ee.Algorithms.If(
                    acc.size().eq(0),
                    cat_one(acc, v),
                    ee.Algorithms.If(
                        v.lte(ensure_number(acc.get(-1))),
                        cat_one(
                            acc,
                            ensure_number(acc.get(-1)).add(1e-8 * (idx.add(1))),
                        ),
                        cat_one(acc, v),
                    ),
                )
            except Exception as e:
                logger.error(
                    "_dedup: FAILED at line=zones.py:_dedup | idx type=%s, prev type=%s, error=%s: %s",
                    type(idx).__name__,
                    type(prev).__name__,
                    type(e).__name__,
                    str(e),
                    exc_info=True,
                )
                raise

        # iterate
        try:
            deduped = ee.Algorithms.If(
                ee.Number(size).eq(0),
                ee.List([]),
                ee.List.sequence(0, ee.Number(size).subtract(1)).iterate(
                    _dedup, ee.List([])
                ),
            )
            result = safe_ee_list(deduped)
            logger.debug("_uniq_sort: iterate completed successfully")
            return result
        except Exception as e:
            logger.error(
                "_uniq_sort: FAILED during iterate | error=%s: %s",
                type(e).__name__,
                str(e),
                exc_info=True,
            )
            raise

    try:
        uniq1 = _uniq_sort(vals1)
        logger.debug("robust_quantile_breaks: _uniq_sort completed")
    except Exception as e:
        logger.error(
            "robust_quantile_breaks: FAILED calling _uniq_sort | error=%s: %s",
            type(e).__name__,
            str(e),
            exc_info=True,
        )
        raise

    # If we still don't have K-1 distinct thresholds, try histogram route
    ok1 = uniq1.size().gte(n_classes - 1)

    def _hist_breaks():
        # compute a reasonably fine histogram
        rng = base.reduceRegion(
            ee.Reducer.minMax(),
            region,
            scale,
            maxPixels=1e9,
            bestEffort=True,
            tileScale=tile_scale,
        )
        vmin = ee.Number(rng.get("NDVI_min"))
        vmax = ee.Number(rng.get("NDVI_max"))
        # if range invalid, abort
        ok_rng = vmin.isFinite().And(vmax.isFinite()).And(vmax.gt(vmin))
        return ee.Algorithms.If(
            ok_rng,
            _hist_quantiles(base, region, vmin, vmax, n_classes, scale, tile_scale),
            ee.List([]),
        )

    def _hist_quantiles(img, region, vmin, vmax, n_classes, scale, tile_scale):
        # 512 bins across observed range
        hist = img.reduceRegion(
            reducer=ee.Reducer.fixedHistogram(vmin, vmax, 512),
            geometry=region,
            scale=scale,
            maxPixels=1e9,
            bestEffort=True,
            tileScale=tile_scale,
        ).get("NDVI")
        hist = ee.Array(hist)  # Nx2 [bin_center, count]
        xs = hist.slice(1, 0, 1).project([0])  # centers
        cs = hist.slice(1, 1, 2).project([0])  # counts
        cdf = cs.cumsum()
        tot = cdf.get([-1])
        # target cumulative counts at the desired quantiles
        qs = ee.List.sequence(1, n_classes - 1).map(
            lambda i: ee.Number(i).divide(n_classes)
        )
        targets = ee.Array(qs).multiply(tot)

        # for each target, find first bin where cdf >= target
        def _qfind(t):
            idx = cdf.gte(ee.Number(t)).argmax().get([0])
            return xs.get([idx])

        brks = targets.toList().map(_qfind)
        # enforce strictly increasing via tiny nudges
        brks = _uniq_sort(brks)
        return brks

    uniq2 = safe_ee_list(ee.Algorithms.If(ok1, uniq1, _hist_breaks()))

    return safe_ee_list(
        ee.Algorithms.If(uniq2.size().gte(n_classes - 1), uniq2, ee.List([]))
    )


def kmeans_classify(
    ndvi_img: ee.Image,
    aoi_geom,
    n_classes: int,
    seed: int = 42,
    sample_scale: int = 10,
    tile_scale: int = 4,
    min_samples: int = 5000,
) -> ee.Image:
    """
    Robust k-means on NDVI:
      - float NDVI + tiny jitter to break exact ties (does not change classing perceptibly)
      - guaranteed sample size (upsample coarser if needed)
      - seeded k-means (stable labels across runs)
      - relabels to contiguous 1..K after clustering
    """

    region = ee.FeatureCollection(aoi_geom).geometry().buffer(5).bounds(1)

    # 1) Ensure float + jitter (breaks ties if NDVI is constant)
    x = ndvi_img.select(["NDVI"]).toFloat()
    eps = ee.Image.random(seed).multiply(1e-6)
    xj = x.add(eps).rename("NDVI")  # still effectively NDVI, just tie-broken

    # 2) Build a sample; if too small, resample at coarser scale to hit min_samples
    def _sample_at(scale):
        return xj.sample(
            region=region,
            scale=scale,
            numPixels=1e7,
            seed=seed,
            geometries=False,
            tileScale=tile_scale,
        )

    samp = _sample_at(sample_scale)
    # Count samples server-side
    samp_count = samp.size()
    need_more = samp_count.lt(min_samples)

    # try a 2x coarser scale if needed
    samp = ee.FeatureCollection(
        ee.Algorithms.If(need_more, _sample_at(sample_scale * 2), samp)
    )
    # and 4x if still needed
    samp_count = samp.size()
    samp = ee.FeatureCollection(
        ee.Algorithms.If(samp_count.lt(min_samples), _sample_at(sample_scale * 4), samp)
    )

    # 3) Fit k-means (wekaKMeans supports seed/maxIterations)
    clusterer = ee.Clusterer.wekaKMeans(
        nClusters=n_classes, seed=seed, maxIterations=200
    ).train(samp, ["NDVI"])

    # 4) Classify the image
    raw_labels = xj.cluster(clusterer).rename("zones_raw")

    # 5) Relabel to contiguous 1..K by sorting cluster means (low NDVI → 1, high → K)
    # Compute mean NDVI per label
    try:
        means = xj.addBands(raw_labels).reduceRegion(
            reducer=ee.Reducer.mean().group(groupField=1, groupName="label"),
            geometry=region,
            scale=sample_scale,
            maxPixels=1e9,
            bestEffort=True,
            tileScale=tile_scale,
        )
        groups = safe_ee_list(
            ee.Dictionary(means.get("groups")).get("groups", ee.List([]))
        )
        logger.debug("kmeans_classify: computed group means")
    except Exception as e:
        logger.error(
            "kmeans_classify: FAILED computing group means | error=%s: %s",
            type(e).__name__,
            str(e),
            exc_info=True,
        )
        raise

    def _pair(g):
        try:
            d = ee.Dictionary(g)
            return safe_ee_list([ee.Number(d.get("label")), ee.Number(d.get("mean"))])
        except Exception as e:
            logger.error(
                "kmeans_classify._pair: FAILED | error=%s: %s",
                type(e).__name__,
                str(e),
                exc_info=True,
            )
            raise

    try:
        pairs = groups.map(_pair)  # [[label, mean], ...]
        pairs_sorted = safe_ee_list(pairs).sort(1)  # ascending NDVI
        orig = pairs_sorted.map(lambda p: safe_ee_list(p).get(0))
        ranks = ee.List.sequence(1, ee.Number(pairs_sorted.size()))
        remap_from = safe_ee_list(orig)
        remap_to = safe_ee_list(ranks)
        logger.debug("kmeans_classify: computed remap lists")
    except Exception as e:
        logger.error(
            "kmeans_classify: FAILED building remap lists | error=%s: %s",
            type(e).__name__,
            str(e),
            exc_info=True,
        )
        raise

    try:
        relabeled = raw_labels.remap(remap_from, remap_to, 1).rename("zones_kmeans")
        logger.debug("kmeans_classify: relabeling completed")
    except Exception as e:
        logger.error(
            "kmeans_classify: FAILED during remap | error=%s: %s",
            type(e).__name__,
            str(e),
            exc_info=True,
        )
        raise

    return relabeled.clip(region)


def _normalise_feature(
    mean_image: ee.Image, geometry: ee.Geometry, name: str
) -> ee.Image:
    def _first_value(result: Mapping[str, object] | object) -> object:
        if isinstance(result, Mapping):
            for value in result.values():
                return value
            return 0
        values = getattr(result, "values", lambda: [])()
        if hasattr(values, "get"):
            return values.get(0, 0)
        try:
            return values[0]
        except (
            TypeError,
            KeyError,
            IndexError,
            AttributeError,
        ):  # pragma: no cover - defensive
            return 0

    stats = mean_image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=DEFAULT_SCALE,
        bestEffort=True,
        tileScale=4,
        maxPixels=gee.MAX_PIXELS,
    )
    std_stats = mean_image.reduceRegion(
        reducer=ee.Reducer.stdDev(),
        geometry=geometry,
        scale=DEFAULT_SCALE,
        bestEffort=True,
        tileScale=4,
        maxPixels=gee.MAX_PIXELS,
    )
    mean_value = ee.Number(_first_value(stats))
    std_value = ee.Number(_first_value(std_stats)).max(1e-6)
    return mean_image.subtract(mean_value).divide(std_value).rename(f"norm_{name}")


def _rank_zones(
    cluster_image: ee.Image, ndvi_mean: ee.Image, geometry: ee.Geometry
) -> ee.Image:
    cluster_band = cluster_image.rename("cluster")
    stats_image = cluster_band.addBands(ndvi_mean.rename("mean_ndvi"))
    grouped = stats_image.reduceRegion(
        reducer=ee.Reducer.mean().group(groupField=0, groupName="cluster"),
        geometry=geometry,
        scale=DEFAULT_SCALE,
        bestEffort=True,
        tileScale=4,
        maxPixels=gee.MAX_PIXELS,
    )
    groups = safe_ee_list(grouped.get("groups", ee.List([])))

    def _cluster_value(item):
        info = ee.Dictionary(item)
        return ee.Dictionary(
            {
                "cluster": ee.Number(info.get("cluster")),
                "mean_ndvi": ee.Number(info.get("mean_ndvi", 0)),
            }
        )

    sorted_groups = groups.map(_cluster_value).sort("mean_ndvi")
    source = safe_ee_list(
        sorted_groups.map(lambda g: ee.Number(ee.Dictionary(g).get("cluster")))
    )
    target = ee.List.sequence(1, source.size())
    ranked = ee.Image(
        ee.Algorithms.If(
            ee.Number(source.size()).gt(0),
            cluster_band.remap(source, target, 0),
            cluster_band,
        )
    )
    return ranked.rename("zone")


def _build_ndvi_feature_images(
    ndvi_images: Sequence[ee.Image] | ee.ImageCollection,
    ndvi_stats: Mapping[str, ee.Image],
) -> dict[str, ee.Image]:
    base_collection = _as_image_collection(ndvi_images)

    def _prepare(image: ee.Image) -> ee.Image:
        return ee.Image(image).select(["NDVI"]).toFloat().rename("NDVI")

    collection = base_collection.map(_prepare)
    percentiles = collection.reduce(ee.Reducer.percentile([10, 90]))
    mean_image = ndvi_stats["mean"].rename("NDVI_mean")
    mask = mean_image.mask()

    p10 = percentiles.select("NDVI_p10").rename("NDVI_p10").updateMask(mask)
    p90 = percentiles.select("NDVI_p90").rename("NDVI_p90").updateMask(mask)
    cv_image = ndvi_stats["cv"].rename("NDVI_cv").updateMask(mask)

    return {
        "mean": mean_image.updateMask(mask),
        "p10": p10,
        "p90": p90,
        "cv": cv_image,
    }


def _train_weka_kmeans_classifier(
    stack: ee.Image,
    geometry: ee.Geometry,
    *,
    n_classes: int,
    sample_size: int,
    seed: int = 42,
) -> ee.Clusterer:
    training = stack.sample(
        region=geometry,
        scale=DEFAULT_SCALE,
        numPixels=sample_size,
        seed=seed,
        tileScale=4,
        geometries=False,
    )
    return ee.Clusterer.wekaKMeans(n_classes).train(training)


def _build_ndvi_kmeans_zones(
    *,
    ndvi_images: Sequence[ee.Image] | ee.ImageCollection,
    ndvi_stats: Mapping[str, ee.Image],
    geometry: ee.Geometry,
    n_classes: int,
    smooth_radius_m: float,
    open_radius_m: float,
    close_radius_m: float,
    min_mapping_unit_ha: float,
    sample_size: int,
    rank_by_mean: bool = True,
) -> tuple[ee.Image, dict[str, ee.Image], CleanupResult]:
    stability = ndvi_stats["stability"]
    feature_images = _build_ndvi_feature_images(ndvi_images, ndvi_stats)
    stack = ee.Image.cat(
        [
            feature_images["mean"],
            feature_images["p10"],
            feature_images["p90"],
            feature_images["cv"],
        ]
    ).updateMask(stability)

    clusterer = _train_weka_kmeans_classifier(
        stack,
        geometry,
        n_classes=n_classes,
        sample_size=sample_size,
    )
    clustered = stack.cluster(clusterer)
    ranked = clustered
    if rank_by_mean:
        ranked = _rank_zones(clustered, feature_images["mean"], geometry)
    ranked = ranked.updateMask(stability)

    cleanup = _apply_cleanup_with_fallback_tracking(
        ranked,
        geometry,
        n_classes=n_classes,
        smooth_radius_m=smooth_radius_m,
        open_radius_m=open_radius_m,
        close_radius_m=close_radius_m,
        min_mapping_unit_ha=min_mapping_unit_ha,
    )
    cleanup.image = cleanup.image.rename("zone")

    return cleanup.image, feature_images, cleanup


def _build_multiindex_zones(
    *,
    ndvi_stats: Mapping[str, ee.Image],
    composites: Sequence[ee.Image],
    geometry: ee.Geometry,
    n_classes: int,
    smooth_radius_m: float,
    open_radius_m: float,
    close_radius_m: float,
    min_mapping_unit_ha: float,
    sample_size: int,
) -> tuple[ee.Image, dict[str, ee.Image], CleanupResult]:
    indices = {
        "NDVI": [
            image.normalizedDifference(["B8", "B4"]).rename("NDVI")
            for image in composites
        ],
        "NDRE": [
            image.normalizedDifference(["B8", "B5"]).rename("NDRE")
            for image in composites
        ],
        "NDMI": [
            image.normalizedDifference(["B8", "B11"]).rename("NDMI")
            for image in composites
        ],
        "BSI": [
            image.expression(
                "((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue))",
                {
                    "swir1": image.select("B11"),
                    "red": image.select("B4"),
                    "nir": image.select("B8"),
                    "blue": image.select("B2"),
                },
            ).rename("BSI")
            for image in composites
        ],
    }

    mean_images: dict[str, ee.Image] = {}
    for name, stack in indices.items():
        collection = ee.ImageCollection(stack)
        mean_images[name] = collection.mean().rename(f"{name}_mean")

    normalised = [
        _normalise_feature(mean_images[name], geometry, name)
        for name in sorted(mean_images)
    ]
    stack = ee.Image.cat(normalised)
    stack = stack.updateMask(ndvi_stats["stability"])

    training = stack.sample(
        region=geometry,
        scale=DEFAULT_SCALE,
        numPixels=sample_size,
        seed=42,
        tileScale=4,
        geometries=False,
    )
    clusterer = ee.Clusterer.wekaKMeans(n_classes).train(training)
    clustered = stack.cluster(clusterer)
    ranked = _rank_zones(clustered, ndvi_stats["mean"], geometry).updateMask(
        ndvi_stats["stability"]
    )
    cleanup = _apply_cleanup_with_fallback_tracking(
        ranked,
        geometry,
        n_classes=n_classes,
        smooth_radius_m=smooth_radius_m,
        open_radius_m=open_radius_m,
        close_radius_m=close_radius_m,
        min_mapping_unit_ha=min_mapping_unit_ha,
    )
    cleanup.image = cleanup.image.rename("zone")

    return cleanup.image, mean_images, cleanup


def _build_multiindex_zones_with_features(
    *,
    ndvi_stats: Mapping[str, ee.Image],
    feature_images: Mapping[str, ee.Image],
    geometry: ee.Geometry,
    n_classes: int,
    smooth_radius_m: float,
    open_radius_m: float,
    close_radius_m: float,
    min_mapping_unit_ha: float,
    sample_size: int,
) -> tuple[ee.Image, dict[str, ee.Image], CleanupResult]:
    stability = ndvi_stats["stability"]
    masked_features: dict[str, ee.Image] = {}
    for name, image in feature_images.items():
        masked_features[name] = image.updateMask(stability)

    normalised = [
        _normalise_feature(masked_features[name], geometry, name)
        for name in sorted(masked_features)
    ]
    stack = ee.Image.cat(normalised).updateMask(stability)

    training = stack.sample(
        region=geometry,
        scale=DEFAULT_SCALE,
        numPixels=sample_size,
        seed=42,
        tileScale=4,
        geometries=False,
    )
    clusterer = ee.Clusterer.wekaKMeans(n_classes).train(training)
    clustered = stack.cluster(clusterer)
    ranked = _rank_zones(clustered, ndvi_stats["mean"], geometry).updateMask(stability)
    cleanup = _apply_cleanup_with_fallback_tracking(
        ranked,
        geometry,
        n_classes=n_classes,
        smooth_radius_m=smooth_radius_m,
        open_radius_m=open_radius_m,
        close_radius_m=close_radius_m,
        min_mapping_unit_ha=min_mapping_unit_ha,
    )
    cleanup.image = cleanup.image.rename("zone")

    return cleanup.image, masked_features, cleanup


def _prepare_selected_period_artifacts(
    aoi_geojson: dict | ee.Geometry,
    *,
    geometry: ee.Geometry,
    working_dir: Path,
    months: Sequence[str],
    start_date: date,
    end_date: date,
    cloud_prob_max: int,
    mask_mode: str,
    n_classes: int,
    cv_mask_threshold: float,
    min_obs_for_cv: int = 3,
    apply_stability_mask: bool | None,
    stability_adaptive: bool = True,
    stability_enforce: bool = False,
    min_valid_ratio: float,
    min_mapping_unit_ha: float,
    smooth_radius_m: float,
    open_radius_m: float,
    close_radius_m: float,
    simplify_tol_m: float,
    simplify_buffer_m: float,
    method: str,
    sample_size: int,
    include_stats: bool,
    debug_dump: bool = False,
    guard: Guard | None = None,
) -> tuple[ZoneArtifacts, dict[str, object]]:
    _ = (
        cv_mask_threshold,
        apply_stability_mask,
        simplify_tol_m,
        simplify_buffer_m,
        method,
        sample_size,
    )
    ee_geometry = geometry
    if not isinstance(geometry, ee.Geometry):
        try:
            ee_geometry = _to_ee_geometry(geometry)
        except Exception:
            ee_geometry = geometry
    geometry = ee_geometry

    diag_guard = guard
    region = None
    if isinstance(geometry, ee.Geometry):
        try:
            region = (
                ee.FeatureCollection([ee.Feature(geometry)])
                .geometry()
                .buffer(5)
                .bounds(1)
            )
        except Exception:
            region = None
    ordered_months = list(_ordered_months(months))
    logger.info("zones:ordered_months=%s", ordered_months)
    if guard is not None:
        guard.record("ordered_months", months=ordered_months)
    per_month_preview: list[object] | None = None
    mask_tier_preview: list[tuple[object, object]] | None = None
    monthly_ndvi_collection: ee.ImageCollection | None = None

    if diag_guard is not None:
        monthly_ndvi_collection = build_monthly_ndvi_collection(
            geometry,
            list(ordered_months),
            min_valid_ratio=min_valid_ratio,
            mask_mode=mask_mode,
            cloud_prob_max=cloud_prob_max,
        )
        try:
            region = ee.FeatureCollection(geometry).geometry().buffer(5).bounds(1)
        except Exception:
            region = geometry

        def _month_diag(img):
            ym = img.get("ym")
            mm = img.reduceRegion(
                ee.Reducer.minMax(),
                region,
                10,
                maxPixels=1e9,
                bestEffort=True,
                tileScale=4,
            )
            vv = img.mask().reduceRegion(
                ee.Reducer.sum(),
                region,
                10,
                maxPixels=1e9,
                bestEffort=True,
                tileScale=4,
            )
            feature = ee.Feature(
                None,
                {
                    "ym": ym,
                    "NDVI_min": mm.get("NDVI_min"),
                    "NDVI_max": mm.get("NDVI_max"),
                    "valid_sum": ee.Number(vv.values().reduce(ee.Reducer.sum())),
                },
            )
            return feature.set("mask_tier", img.get("mask_tier"))

        _per_month = ee.FeatureCollection(monthly_ndvi_collection.map(_month_diag))
        try:
            per_month_py = _per_month.aggregate_array("properties").getInfo()
        except Exception:
            per_month_py = None
        try:
            diag_guard.record("ndvi_per_month", items=per_month_py)
        except Exception:
            pass
        if isinstance(per_month_py, list):
            per_month_preview = per_month_py[:3]
        try:
            ym_list = monthly_ndvi_collection.aggregate_array("ym").getInfo()
            tier_list = monthly_ndvi_collection.aggregate_array("mask_tier").getInfo()
            if isinstance(ym_list, list) and isinstance(tier_list, list):
                mask_tier_preview = list(zip(ym_list, tier_list))
        except Exception:
            pass

        # After server-side NDVI filter
        col_size = monthly_ndvi_collection.size().getInfo()
        diag_guard.record("monthly_stack_size_after_filter", images=col_size)
        diag_guard.require(
            col_size and col_size > 0,
            "E_NO_MONTHS",
            "No monthly NDVI composites contain the NDVI band.",
            "Relax mask or widen date range.",
        )

        first_image = ee.Image(monthly_ndvi_collection.first())
        first_band_names = first_image.bandNames().getInfo() or []
        first_ym = first_image.get("ym").getInfo() if first_image.get("ym") else None
        diag_guard.record("first_valid_month", ym=first_ym, bands=first_band_names)

        diag_guard.require(
            "NDVI" in first_band_names,
            "E_MEAN_EMPTY",
            "First valid monthly composite lacks NDVI band.",
            "Ensure compute_ndvi() returns 'NDVI' and masks aren't over-aggressive.",
        )
    composites, skipped_months, composite_metadata = _build_composite_series(
        geometry,
        ordered_months,
        start_date,
        end_date,
        cloud_prob_max,
        mask_mode,
        min_valid_ratio,
    )
    if diag_guard is not None:
        image_count = len(composites)
        diag_guard.record("monthly_stack", images=image_count)
        logger.info("zones:monthly_stack images=%s", image_count)
        diag_guard.require(
            image_count > 0,
            "E_NO_MONTHS",
            "No monthly composites built.",
            "Widen dates; relax clouds.",
            images=image_count,
        )
    if not composites:
        raise ValueError(
            "No valid Sentinel-2 scenes were found for the selected months"
        )

    composite_images = [image for _, image in composites]

    if diag_guard is not None and composite_images:
        try:
            first_image = _compute_ndvi(composite_images[0])
            first_stats = first_image.reduceRegion(
                reducer=ee.Reducer.minMax(),
                geometry=geometry,
                scale=DEFAULT_SCALE,
                maxPixels=gee.MAX_PIXELS,
                bestEffort=True,
                tileScale=4,
            )
            first_info = (
                first_stats.getInfo()
                if hasattr(first_stats, "getInfo")
                else first_stats
            )
        except Exception:
            first_info = {}
        ndvi_min = None
        ndvi_max = None
        if isinstance(first_info, Mapping):
            ndvi_min = first_info.get("NDVI_min")
            ndvi_max = first_info.get("NDVI_max")
            if ndvi_min is None:
                ndvi_min = first_info.get("NDVI")
            if ndvi_max is None:
                ndvi_max = first_info.get("NDVI")
        diag_guard.record("first_month_ndvi", ndvi_min=ndvi_min, ndvi_max=ndvi_max)
        logger.info("zones:first_month_ndvi min=%s max=%s", ndvi_min, ndvi_max)
        diag_guard.require(
            ndvi_min is not None and ndvi_max is not None,
            "E_FIRST_MONTH_EMPTY",
            "First month NDVI has no valid pixels.",
            "Relax cloud mask.",
            ndvi_min=ndvi_min,
            ndvi_max=ndvi_max,
        )

    ndvi_collection: ee.ImageCollection | None = None
    try:
        base_collection = ee.ImageCollection(composite_images)
        ndvi_collection = base_collection.map(_compute_ndvi)
        ndvi_images: Sequence[ee.Image] | ee.ImageCollection = ndvi_collection
    except Exception:
        ndvi_collection = None
        ndvi_images = [_compute_ndvi(image) for image in composite_images]

    ndvi_stats = dict(_ndvi_temporal_stats(ndvi_images))
    try:
        mean_band = ee.String(ndvi_stats["mean"].bandNames().get(0))
        ndvi_stats["mean"] = (
            ee.Image(ndvi_stats["mean"]).select([mean_band]).rename("NDVI")
        )
    except Exception:
        ndvi_stats["mean"] = ee.Image(ndvi_stats["mean"]).rename("NDVI")

    if ndvi_collection is not None:
        try:
            valid_mask = ndvi_collection.count().gt(0)
            mean_image = (
                _ndvi_1band(ndvi_collection.mean()).toFloat().updateMask(valid_mask)
            )
            neighbors = mean_image.focal_mean(radius=30, units="meters")
            fill_source = neighbors
            try:
                hole_mask = mean_image.mask().Not()
                small_holes = (
                    hole_mask.connectedPixelCount(maxSize=9, eightConnected=True)
                    .gt(0)
                    .selfMask()
                )
                fill_source = neighbors.updateMask(small_holes)
            except Exception:  # pragma: no cover - logging guard
                logger.exception(
                    "Failed to apply NDVI hole size mask; using full neighbors"
                )
            mean_image = mean_image.unmask(fill_source)
            mean_image = mean_image.clip(geometry)
            ndvi_stats["mean"] = mean_image
            ndvi_stats["valid_mask"] = valid_mask
        except Exception:  # pragma: no cover - logging guard
            logger.exception("Failed to compute mean NDVI image from collection")
    else:
        try:
            ndvi_stats["mean"] = ee.Image(ndvi_stats["mean"]).toFloat().clip(geometry)
        except Exception:  # pragma: no cover - logging guard
            logger.exception("Failed to clip fallback NDVI mean image")

    if diag_guard is not None and monthly_ndvi_collection is not None:
        region = ee.FeatureCollection(geometry).geometry().buffer(5).bounds(1)
        valid_mask_sum = (
            monthly_ndvi_collection.map(lambda im: _ndvi_1band(ee.Image(im)).mask())
            .sum()
            .reduceRegion(
                ee.Reducer.sum(),
                region,
                10,
                maxPixels=1e9,
                bestEffort=True,
                tileScale=4,
            )
            .getInfo()
        )
        diag_guard.record("pre_mean_valid_mask_sum", total=valid_mask_sum)
        diag_guard.require(
            valid_mask_sum
            and any(
                isinstance(v, (int, float)) and v > 0
                for v in (
                    valid_mask_sum.values()
                    if isinstance(valid_mask_sum, dict)
                    else [valid_mask_sum]
                )
            ),
            "E_MEAN_EMPTY",
            "All NDVI pixels masked before mean().",
            "Relax cloud mask; verify date range; ensure AOI intersects imagery.",
            mask_sum=valid_mask_sum,
        )
        collection_for_stats = monthly_ndvi_collection or ndvi_collection
        if collection_for_stats is None:
            try:
                collection_for_stats = ee.ImageCollection(ndvi_images)
            except Exception:
                collection_for_stats = None
        ndvi_mean_image = None
        if collection_for_stats is not None:
            try:
                ndvi_mean_image = _ndvi_1band(collection_for_stats.mean())
            except Exception:
                ndvi_mean_image = None
        stats_info = {}
        if ndvi_mean_image is not None:
            try:
                stats_info = ndvi_mean_image.reduceRegion(
                    reducer=ee.Reducer.minMax(),
                    geometry=region,
                    scale=10,
                    maxPixels=1e9,
                    bestEffort=True,
                    tileScale=4,
                ).getInfo()
            except Exception:
                stats_info = {}
        mean_min = stats_info.get("NDVI_min") if isinstance(stats_info, dict) else None
        mean_max = stats_info.get("NDVI_max") if isinstance(stats_info, dict) else None
        diag_guard.record("ndvi_stats", mean_min=mean_min, mean_max=mean_max)
        logger.info("zones:mean_ndvi min=%s max=%s", mean_min, mean_max)
        context_kwargs = stats_info if isinstance(stats_info, dict) else {}
        diag_guard.require(
            mean_min is not None and mean_max is not None,
            "E_MEAN_EMPTY",
            "Mean NDVI reduction returned no values.",
            "Likely over-masking or geometry/scale issue. Try bestEffort/tileScale, relax cloud mask.",
            **context_kwargs,
        )
        if mean_min is not None and mean_max is not None:
            diag_guard.require(
                mean_min < mean_max,
                "E_MEAN_CONSTANT",
                "Mean NDVI is constant across AOI.",
                "Fix NDVI bands/float; ensure variability; check masks.",
                **context_kwargs,
            )

    diag_region = geometry
    if ndvi_collection is not None:
        try:
            size_value = int(ee.Number(ndvi_collection.size()).getInfo() or 0)
            logger.info("Zone NDVI collection size: %s", size_value)
        except Exception:  # pragma: no cover - logging guard
            logger.exception("Failed to evaluate NDVI collection size for zone export")

    diag_kwargs = {
        "geometry": diag_region,
        "scale": DEFAULT_SCALE,
        "bestEffort": True,
        "tileScale": 4,
        "maxPixels": gee.MAX_PIXELS,
    }

    try:
        raw_count = ndvi_stats["mean"].mask().reduce(ee.Reducer.count())
        raw_stats = raw_count.reduceRegion(reducer=ee.Reducer.sum(), **diag_kwargs)
        raw_info = raw_stats.getInfo() if hasattr(raw_stats, "getInfo") else raw_stats
        logger.info("Zone NDVI raw pixel stats: %s", raw_info)
    except Exception:  # pragma: no cover - logging guard
        logger.exception("Failed to compute NDVI raw pixel statistics")

    try:
        minmax = ndvi_stats["mean"].reduceRegion(
            reducer=ee.Reducer.minMax(),
            **diag_kwargs,
        )
        minmax_info = minmax.getInfo() if hasattr(minmax, "getInfo") else minmax
        logger.info("Zone NDVI min/max: %s", minmax_info)
        print("Zone NDVI min/max:", minmax_info)
    except Exception:  # pragma: no cover - logging guard
        logger.exception("Failed to compute NDVI min/max diagnostics")

    try:
        histogram = ndvi_stats["mean"].reduceRegion(
            reducer=ee.Reducer.histogram(),
            **diag_kwargs,
        )
        histogram_info = (
            histogram.getInfo() if hasattr(histogram, "getInfo") else histogram
        )
        logger.info("Zone NDVI histogram: %s", histogram_info)
    except Exception:  # pragma: no cover - logging guard
        logger.exception("Failed to compute NDVI histogram diagnostics")

    try:
        percentiles = ndvi_stats["mean"].reduceRegion(
            reducer=ee.Reducer.percentile([5, 25, 50, 75, 95]),
            **diag_kwargs,
        )
        percentile_info = (
            percentiles.getInfo() if hasattr(percentiles, "getInfo") else percentiles
        )
        logger.info("Zone NDVI percentiles: %s", percentile_info)
        print("Zone NDVI percentiles:", percentile_info)
    except Exception:  # pragma: no cover - logging guard
        logger.exception("Failed to compute NDVI percentile diagnostics")

    try:
        region = ee.FeatureCollection(aoi_geojson).geometry().buffer(5).bounds(1)
    except Exception:
        try:
            region = (
                ee.FeatureCollection([ee.Feature(geometry)])
                .geometry()
                .buffer(5)
                .bounds(1)
            )
        except Exception:
            region = geometry

    ndvi_stack = monthly_ndvi_collection or ndvi_collection
    if ndvi_stack is None:
        try:
            ndvi_stack = ee.ImageCollection(ndvi_images)
        except Exception:
            ndvi_stack = None

    ndvi_mean: ee.Image | None = None
    if monthly_ndvi_collection is not None:
        try:
            ndvi_mean = _ndvi_1band(monthly_ndvi_collection.mean())
        except Exception:
            ndvi_mean = None
    if ndvi_mean is None and ndvi_stack is not None:
        try:
            ndvi_mean = _ndvi_1band(ndvi_stack.mean())
        except Exception:
            ndvi_mean = None
    if ndvi_mean is None:
        ndvi_mean = _ndvi_1band(ee.Image(ndvi_stats["mean"]))

    try:
        region = ee.FeatureCollection(aoi_geojson).geometry().buffer(5).bounds(1)
    except Exception:
        try:
            region = (
                ee.FeatureCollection([ee.Feature(geometry)])
                .geometry()
                .buffer(5)
                .bounds(1)
            )
        except Exception:
            region = geometry

    common_mask = ee.Image(1)
    if monthly_ndvi_collection is not None:
        try:
            count_img = monthly_ndvi_collection.reduce(ee.Reducer.count()).select(
                "NDVI_count"
            )
            common_mask = count_img.gte(min_obs_for_cv)
        except Exception:
            common_mask = ee.Image(1)

    ndvi_mean = normalize_ndvi_band(ee.Image(ndvi_mean))
    if guard is not None:
        try:
            guard.record(
                "ndvi_band_norm",
                band_names=ndvi_mean.bandNames().getInfo(),
            )
        except Exception:
            pass
    ndvi_mean = ndvi_mean.updateMask(common_mask)

    def _coverage_ratio(img, region, scale=10, tile_scale=4):
        try:
            return coverage_ratio(img, region, scale=scale, tile_scale=tile_scale)
        except Exception:
            return _cover_ratio(img, region, scale=scale, tile_scale=tile_scale)

    pre_cov = _coverage_ratio(ndvi_mean, region)
    pre_cov_value: float | None = None
    try:
        pre_cov_value = float(pre_cov.getInfo())
    except Exception:
        pre_cov_value = None

    if guard is not None:
        try:
            guard.record(
                "coverage_before_stability",
                valid_ratio=pre_cov_value,
            )
        except Exception:
            pass

    mask_tier_preview_payload = None
    if mask_tier_preview:
        mask_tier_preview_payload = [
            {"ym": ym, "mask_tier": tier} for ym, tier in list(mask_tier_preview)[:6]
        ]
    coverage_ctx = {
        "valid_ratio": pre_cov_value,
        "min_valid_ratio": float(min_valid_ratio),
        "per_month_preview": per_month_preview,
        "mask_tiers": mask_tier_preview_payload,
    }
    coverage_ok = pre_cov_value is not None and pre_cov_value >= float(min_valid_ratio)
    if guard is not None:
        guard.require(
            coverage_ok,
            "E_COVERAGE_LOW",
            f"Too few valid pixels before stability (valid_ratio < {min_valid_ratio}).",
            "Relax masks or widen the month range.",
            **coverage_ctx,
        )
    elif not coverage_ok:
        raise PipelineError(
            code="E_COVERAGE_LOW",
            msg=(
                "E_COVERAGE_LOW: Too few valid pixels before stability "
                f"(valid_ratio < {min_valid_ratio})."
            ),
            hints="Relax masks or widen the month range.",
            ctx=coverage_ctx,
        )

    stability_flag = (
        APPLY_STABILITY if apply_stability_mask is None else bool(apply_stability_mask)
    )
    mean_before_stability = ndvi_mean
    stability_image = ee.Image(1)
    masked_mean = ndvi_mean
    post_cov = pre_cov
    post_cov_value: float | None = pre_cov_value
    stability_applied_bool: bool | None = False
    stability_applied_reason: str | None = None

    if stability_flag and monthly_ndvi_collection is not None:
        stable_region = region or geometry
        monthly_norm = monthly_ndvi_collection.map(normalize_ndvi_band)
        stats_image = stats_stack(monthly_norm).updateMask(common_mask)
        ndvi_cv = stats_image.select("NDVI_cv").updateMask(common_mask)
        thresholds = ensure_list(cv_mask_threshold).cat(
            ee.List(STABILITY_THRESHOLD_SEQUENCE)
        )
        stability_candidate = stability_mask_from_cv(
            ndvi_cv,
            stable_region,
            thresholds,
            scale=DEFAULT_SCALE,
            tile_scale=4,
            min_survival_ratio=MIN_STABILITY_SURVIVAL_RATIO,
        )

        masked_candidate = ndvi_mean.updateMask(stability_candidate)

        post_cov = _coverage_ratio(masked_candidate, stable_region)
        try:
            post_cov_value = float(post_cov.getInfo())
        except Exception:
            post_cov_value = None

        if guard is not None:
            try:
                cv_stats = ndvi_cv.reduceRegion(
                    ee.Reducer.minMax().combine(ee.Reducer.mean(), "", True),
                    stable_region,
                    10,
                    maxPixels=1e9,
                    bestEffort=True,
                    tileScale=4,
                )
                info = cv_stats.getInfo() if hasattr(cv_stats, "getInfo") else cv_stats
                if isinstance(info, Mapping):
                    guard.record(
                        "stability_cv_stats",
                        **{
                            k: float(v)
                            for k, v in info.items()
                            if isinstance(v, (int, float))
                        },
                    )
                guard.record(
                    "coverage_after_stability",
                    valid_ratio=post_cov_value,
                )
                guard.record("min_obs_for_cv", value=min_obs_for_cv)
            except Exception:
                pass

        if stability_adaptive:
            masked_mean = ee.Image(
                ee.Algorithms.If(
                    post_cov.gte(min_valid_ratio),
                    masked_candidate,
                    ndvi_mean,
                )
            )
            stability_image = ee.Image(
                ee.Algorithms.If(
                    post_cov.gte(min_valid_ratio),
                    stability_candidate,
                    ee.Image(1),
                )
            )
            try:
                applied = bool(post_cov.gte(min_valid_ratio).getInfo())
            except Exception:
                applied = post_cov_value is not None and post_cov_value >= float(
                    min_valid_ratio
                )
            stability_applied_bool = applied
            stability_applied_reason = "ok" if applied else "coverage_drop_bypass"
            if guard is not None:
                try:
                    guard.record(
                        "stability_applied",
                        applied=applied,
                        reason=stability_applied_reason,
                    )
                except Exception:
                    pass
        else:
            post_ok = post_cov_value is not None and post_cov_value >= float(
                min_valid_ratio
            )
            if stability_enforce:
                if guard is not None:
                    guard.require(
                        post_ok,
                        "E_STABILITY_EMPTY",
                        "All pixels removed by stability mask.",
                        "Increase cv_mask_threshold, widen months, or disable stability.",
                    )
                elif not post_ok:
                    raise PipelineError(
                        code="E_STABILITY_EMPTY",
                        msg="E_STABILITY_EMPTY: All pixels removed by stability mask.",
                        hints="Increase cv_mask_threshold, widen months, or disable stability.",
                    )
            masked_mean = masked_candidate
            stability_image = stability_candidate
            stability_applied_bool = True
            stability_applied_reason = "forced"
            if guard is not None:
                try:
                    guard.record(
                        "stability_applied",
                        applied=True,
                        reason=stability_applied_reason,
                    )
                except Exception:
                    pass
    elif stability_flag:
        stability_applied_bool = False
        stability_applied_reason = "no_monthly_collection"
        stability_image = ee.Image(1)
        masked_mean = ndvi_mean
        if guard is not None:
            try:
                guard.record(
                    "stability_applied",
                    applied=False,
                    reason=stability_applied_reason,
                )
            except Exception:
                pass

    if guard is not None and not (
        stability_flag and monthly_ndvi_collection is not None
    ):
        try:
            guard.record(
                "coverage_after_stability",
                valid_ratio=post_cov_value,
            )
        except Exception:
            pass
        try:
            guard.record(
                "stability_applied",
                applied=bool(stability_applied_bool),
                reason=stability_applied_reason,
            )
        except Exception:
            pass

    masked_mean = normalize_ndvi_band(ee.Image(masked_mean))

    ndvi_stats["stability"] = stability_image
    ndvi_stats["mean"] = masked_mean
    mean_float = _ensure_float_image(masked_mean)
    if diag_guard is not None:

        def _extract_count(payload: Mapping[str, object] | None) -> int:
            if not isinstance(payload, Mapping):
                return 0
            for key in ("NDVI", "NDVI_mask"):
                value = payload.get(key)
                if value is not None:
                    try:
                        return int(value)
                    except Exception:
                        continue
            values = [payload.get(key) for key in payload]
            for candidate in values:
                if candidate is not None:
                    try:
                        return int(candidate)
                    except Exception:
                        continue
            return 0

        vb_cnt = 0
        va_cnt = 0
        try:
            base_mask = mean_before_stability.mask()
            vb_stats = base_mask.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=geometry,
                scale=DEFAULT_SCALE,
                maxPixels=gee.MAX_PIXELS,
                bestEffort=True,
                tileScale=4,
            )
            vb_info = vb_stats.getInfo() if hasattr(vb_stats, "getInfo") else vb_stats
            vb_cnt = _extract_count(vb_info)
        except Exception:
            vb_cnt = 0
        try:
            stable_mask = mean_before_stability.updateMask(stability_image).mask()
            va_stats = stable_mask.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=geometry,
                scale=DEFAULT_SCALE,
                maxPixels=gee.MAX_PIXELS,
                bestEffort=True,
                tileScale=4,
            )
            va_info = va_stats.getInfo() if hasattr(va_stats, "getInfo") else va_stats
            va_cnt = _extract_count(va_info)
        except Exception:
            va_cnt = 0
        diag_guard.record(
            "stability",
            cv_thresh=cv_mask_threshold,
            valid_before=vb_cnt,
            valid_after=va_cnt,
        )
        logger.info(
            "zones:stability cv=%.3f before=%s after=%s",
            cv_mask_threshold,
            vb_cnt,
            va_cnt,
        )
        enforce_empty = (
            stability_flag
            and (not stability_adaptive or bool(stability_applied_bool))
            and stability_enforce
        )
        if enforce_empty:
            diag_guard.require(
                va_cnt > 0,
                "E_STABILITY_EMPTY",
                "All pixels removed by stability mask.",
                "Increase cv_mask_threshold, widen months, or disable stability.",
                before=vb_cnt,
                after=va_cnt,
            )

        try:
            rng = masked_mean.reduceRegion(
                ee.Reducer.minMax(),
                geometry,
                10,
                maxPixels=1e9,
                bestEffort=True,
                tileScale=4,
            ).getInfo()
        except Exception:
            rng = None

        try:
            std = masked_mean.reduceRegion(
                ee.Reducer.stdDev(),
                geometry,
                10,
                maxPixels=1e9,
                bestEffort=True,
                tileScale=4,
            ).getInfo()
        except Exception:
            std = None

        diag_guard.record(
            "ndvi_distribution",
            min=rng.get("NDVI_min") if isinstance(rng, Mapping) else None,
            max=rng.get("NDVI_max") if isinstance(rng, Mapping) else None,
            std=std.get("NDVI_stdDev") if isinstance(std, Mapping) else None,
        )

    mean_image = mean_float.rename("NDVI_mean")

    workdir = _ensure_working_directory(working_dir)
    ndvi_path = workdir / "mean_ndvi.tif"
    mean_export = _download_image_to_path(mean_image, geometry, ndvi_path)
    ndvi_path = mean_export.path

    mmu_value = max(min_mapping_unit_ha, 0)
    mmu_applied = True
    if isinstance(aoi_geojson, dict):
        try:
            if area_ha(aoi_geojson) < min_mapping_unit_ha:
                mmu_value = 0
                mmu_applied = False
        except Exception:  # pragma: no cover
            mmu_value = max(min_mapping_unit_ha, 0)
            mmu_applied = True

    def _guard_require(cond: bool, code: str, msg: str, hints: str, **ctx) -> None:
        if guard is not None:
            guard.require(cond, code, msg, hints, **ctx)
        elif not cond:
            raise PipelineError(code=code, msg=f"{code}: {msg}", hints=hints, ctx=ctx)

    valid_methods = {"ndvi_kmeans", "ndvi_percentiles", "multiindex_kmeans"}
    if method not in valid_methods:
        raise PipelineError("E_METHOD_UNKNOWN", f"Unknown method: {method}")

    if method == "ndvi_kmeans":
        masked_mean = _ndvi_1band(ee.Image(masked_mean))
        g = guard if guard is not None else Guard()
        feature_payload: dict[str, ee.Image] = {"NDVI": masked_mean}

        if diag_guard is not None:
            diag_guard.record(
                "classification", method="kmeans", seed=42, n_classes=n_classes
            )
            logger.info("zones:classification method=%s", method)
            logger.info("zones:classification kmeans seed=42")

        region_for_kmeans = region
        if region_for_kmeans is None and isinstance(geometry, ee.Geometry):
            try:
                region_for_kmeans = (
                    ee.FeatureCollection([ee.Feature(geometry)])
                    .geometry()
                    .buffer(5)
                    .bounds(1)
                )
            except Exception:
                region_for_kmeans = geometry
        if region_for_kmeans is None:
            region_for_kmeans = geometry

        bands = masked_mean.bandNames()
        mask_bands = masked_mean.mask().bandNames().size()

        valid_sum = ee.Dictionary(
            masked_mean.mask().reduceRegion(
                ee.Reducer.sum(),
                region_for_kmeans,
                10,
                maxPixels=1e9,
                bestEffort=True,
                tileScale=4,
            )
        )
        total_px = ee.Number(region_for_kmeans.area(1)).divide(ee.Number(10).pow(2))
        safe_total_px = ee.Number(
            ee.Algorithms.If(total_px.gt(0), total_px, ee.Number(1))
        )
        valid_ratio = ee.Number(valid_sum.values().reduce(ee.Reducer.sum())).divide(
            safe_total_px
        )

        rng = ee.Dictionary(
            masked_mean.reduceRegion(
                ee.Reducer.minMax(),
                region_for_kmeans,
                10,
                maxPixels=1e9,
                bestEffort=True,
                tileScale=4,
            )
        )
        vmin = ee.Number(rng.get("NDVI_min", ee.Number(0)))
        vmax = ee.Number(rng.get("NDVI_max", ee.Number(0)))

        valid_ratio_val: float | None = None
        ndvi_min_val: float | None = None
        ndvi_max_val: float | None = None
        try:
            valid_ratio_val = float(valid_ratio.getInfo())
        except Exception:
            valid_ratio_val = None
        try:
            ndvi_min_val = float(vmin.getInfo())
        except Exception:
            ndvi_min_val = None
        try:
            ndvi_max_val = float(vmax.getInfo())
        except Exception:
            ndvi_max_val = None

        try:
            g.record(
                "ndvi_input_health",
                band_names=bands.getInfo(),
                mask_band_count=int(mask_bands.getInfo()),
                valid_ratio=valid_ratio_val,
                ndvi_min=ndvi_min_val,
                ndvi_max=ndvi_max_val,
            )
        except Exception:
            pass

        g.require(
            bands.size().eq(1).And(bands.contains("NDVI")),
            "E_NDVI_BAND",
            "Classifier input must be a single band named 'NDVI'.",
            "Ensure compute_ndvi() renames and monthly composites select 'NDVI'; avoid visualize().",
        )
        g.require(
            mask_bands.eq(1),
            "E_MASK_SHAPE",
            "NDVI mask must be single-band.",
            "Use B8.mask().And(B4.mask()) in compute_ndvi().",
        )
        g.require(
            valid_ratio.gte(0.25),
            "E_COVERAGE_LOW",
            "Too few valid pixels after masking (valid_ratio < 0.25).",
            "Relax SCL/cloud masks or widen month range.",
        )
        g.require(
            vmax.gt(vmin),
            "E_RANGE_EMPTY",
            (
                f"NDVI has no dynamic range (min == max). "
                f"NDVI_min={ndvi_min_val}, NDVI_max={ndvi_max_val}."
            ),
            "Check NDVI bands/float math; relax masks; widen months; verify AOI intersects imagery; do NOT use visualize().",
        )

        if diag_guard is not None:
            try:
                rng_info = masked_mean.reduceRegion(
                    ee.Reducer.minMax(),
                    geometry,
                    10,
                    maxPixels=1e9,
                    bestEffort=True,
                    tileScale=4,
                ).getInfo()
            except Exception:
                rng_info = None

            try:
                std_info = masked_mean.reduceRegion(
                    ee.Reducer.stdDev(),
                    geometry,
                    10,
                    maxPixels=1e9,
                    bestEffort=True,
                    tileScale=4,
                ).getInfo()
            except Exception:
                std_info = None

            diag_guard.record(
                "ndvi_distribution_for_kmeans",
                min=rng_info.get("NDVI_min") if isinstance(rng_info, Mapping) else None,
                max=rng_info.get("NDVI_max") if isinstance(rng_info, Mapping) else None,
                std=(
                    std_info.get("NDVI_stdDev")
                    if isinstance(std_info, Mapping)
                    else None
                ),
            )

        eps = ee.Image.random(42).multiply(1e-6)
        ndvi_for_kmeans = (
            normalize_ndvi_band(masked_mean.toFloat()).add(eps).rename("NDVI")
        )

        zones_raw = kmeans_classify(
            ndvi_for_kmeans,
            geometry,
            n_classes=n_classes,
            seed=42,
            sample_scale=10,
            tile_scale=4,
            min_samples=int(max(sample_size, 0)),
        )

        if diag_guard is not None:
            try:
                lbl_hist = zones_raw.reduceRegion(
                    ee.Reducer.frequencyHistogram(),
                    geometry,
                    10,
                    maxPixels=1e9,
                    bestEffort=True,
                    tileScale=4,
                ).getInfo()
            except Exception:
                lbl_hist = None
            diag_guard.record("kmeans_label_histogram", hist=lbl_hist)

        zones_raster = zones_raw.rename("zone")
        cleanup_result = _apply_cleanup_with_fallback_tracking(
            zones_raster,
            geometry,
            n_classes=n_classes,
            smooth_radius_m=smooth_radius_m,
            open_radius_m=open_radius_m,
            close_radius_m=close_radius_m,
            min_mapping_unit_ha=mmu_value,
        )
        cleanup_result.image = cleanup_result.image.rename("zone")
        zone_image = cleanup_result.image

        zone_raster_path = workdir / "zones_classified.tif"
        zone_export = _download_image_to_path(zone_image, geometry, zone_raster_path)
        zone_raster_path = zone_export.path

        with rasterio.open(zone_raster_path) as zone_src:
            classified_data = zone_src.read(1)
            zone_transform = zone_src.transform
            zone_crs = zone_src.crs

        with rasterio.open(ndvi_path) as ndvi_src:
            ndvi_values = ndvi_src.read(1, masked=True)

        smoothing_requested = {
            "smooth_radius_m": float(max(smooth_radius_m, 0)),
            "open_radius_m": float(max(open_radius_m, 0)),
            "close_radius_m": float(max(close_radius_m, 0)),
            "min_mapping_unit_ha": float(max(mmu_value, 0)),
        }
        applied_operations = dict(cleanup_result.applied_operations)
        executed_operations = dict(cleanup_result.executed_operations)
        mmu_was_applied = bool(applied_operations.get("min_mapping_unit"))

        populated = classified_data[classified_data > 0]
        final_zone_count = int(np.unique(populated).size) if populated.size else 0
        effective_zone_count = (
            min(n_classes, final_zone_count) if final_zone_count else 0
        )

        artifacts, local_metadata = _assemble_zone_artifacts(
            classified=classified_data,
            ndvi=ndvi_values,
            transform=zone_transform,
            crs=zone_crs,
            working_dir=workdir,
            include_stats=include_stats,
            raster_path=zone_raster_path,
            mean_ndvi_path=ndvi_path,
            smoothing_requested=smoothing_requested,
            applied_operations=applied_operations,
            executed_operations=executed_operations,
            fallback_applied=bool(cleanup_result.fallback_applied),
            fallback_removed=list(cleanup_result.fallback_removed),
            min_mapping_unit_ha=mmu_value,
            requested_zone_count=n_classes,
            effective_zone_count=effective_zone_count,
            classification_method="ndvi_kmeans",
            thresholds=[],
            kmeans_fallback_applied=False,
            kmeans_cluster_centers=[],
        )

        feature_names = sorted(str(name) for name in feature_payload.keys())
        feature_metadata = {"method": "kmeans", "features": feature_names}

        metadata = {
            "used_months": ordered_months,
            "skipped_months": skipped_months,
            "min_mapping_unit_applied": mmu_was_applied,
            "mmu_applied": mmu_was_applied,
            "zone_method": method,
            "kmeans_features": feature_metadata,
            "kmeans_feature_count": len(feature_names),
            "kmeans_sample_size": int(sample_size),
            "stability_thresholds": list(STABILITY_THRESHOLD_SEQUENCE),
            "stability_mask_applied": stability_flag,
        }

        metadata.update(composite_metadata)
        metadata.update(local_metadata)
        metadata["downloaded_mean_ndvi"] = str(ndvi_path)
        metadata["downloaded_zone_raster"] = str(zone_raster_path)
        metadata["mean_ndvi_export_task"] = _task_payload(mean_export.task)
        metadata["zone_raster_export_task"] = _task_payload(zone_export.task)

        area_stats = (
            local_metadata.get("zones") if isinstance(local_metadata, Mapping) else None
        )
        area_list = area_stats if isinstance(area_stats, list) else []
        if diag_guard is not None:
            diag_guard.record(
                "cartography",
                open_m=open_radius_m,
                close_m=close_radius_m,
                mmu_ha=mmu_value,
                area_stats=area_stats,
            )
            logger.info(
                "zones:cartography open=%s close=%s mmu_ha=%s polys=%s",
                open_radius_m,
                close_radius_m,
                mmu_value,
                len(area_list),
            )
            diag_guard.require(
                bool(area_list),
                "E_VECT_EMPTY",
                "Vectorization produced no polygons.",
                "Loosen MMU; reduce smoothing.",
                area_stats=area_stats,
            )

        return artifacts, metadata

    if method == "ndvi_percentiles":
        percentile_source = masked_mean
        try:
            percentile_source = masked_mean.select([0]).rename("NDVI")
        except Exception:  # pragma: no cover - defensive guard
            try:
                percentile_source = masked_mean.rename("NDVI")
            except Exception:  # pragma: no cover - defensive guard
                percentile_source = masked_mean

        brks_py = None
        try:
            breaks = robust_quantile_breaks(percentile_source, geometry, n_classes)
            logger.debug(
                "Robust breaks raw value (type=%s): %s",
                type(breaks).__name__,
                breaks,
            )
            if hasattr(breaks, "getInfo"):
                brks_py = breaks.getInfo()
            elif isinstance(breaks, (list, tuple)):
                brks_py = list(breaks)
            else:
                if isinstance(breaks, (int, float)):
                    brks_py = [float(breaks)]
                elif isinstance(breaks, str):
                    brks_py = [breaks]
                else:
                    brks_py = safe_ee_list(breaks).getInfo()
        except Exception:
            brks_py = None

        if diag_guard is not None:
            diag_guard.record("classification", method="percentiles", breaks=brks_py)
        else:
            logger.info("zones:classification method=%s", method)

        required_breaks = max(n_classes - 1, 0)
        has_breaks = isinstance(brks_py, list) and len(brks_py) == required_breaks
        _guard_require(
            bool(has_breaks),
            "E_BREAKS_COLLAPSED",
            "Percentile thresholds collapsed; NDVI spread too small.",
            "Widen date range or use method=ndvi_kmeans.",
            breaks=brks_py,
        )

        forced_thresholds = sorted(float(value) for value in brks_py[:required_breaks])

        artifacts, local_metadata = _classify_local_zones(
            ndvi_path,
            working_dir=workdir,
            n_classes=n_classes,
            min_mapping_unit_ha=mmu_value,
            smooth_radius_m=smooth_radius_m,
            open_radius_m=open_radius_m,
            close_radius_m=close_radius_m,
            include_stats=include_stats,
            forced_thresholds=forced_thresholds,
        )

        percentile_breaks: list[float] | None = None
        raw_breaks = local_metadata.get("percentile_thresholds")
        if isinstance(raw_breaks, Sequence):
            percentile_breaks = [float(value) for value in raw_breaks]
        if diag_guard is not None:
            diag_guard.record(
                "classification_result",
                method=local_metadata.get("classification_method", method),
                breaks=percentile_breaks,
            )
        logger.info("zones:classification method=%s", method)

        mmu_was_applied = mmu_value > 0 and mmu_applied
        metadata: dict[str, object] = {
            "used_months": ordered_months,
            "skipped_months": skipped_months,
            "min_mapping_unit_applied": mmu_was_applied,
            "mmu_applied": mmu_was_applied,
            "zone_method": method,
            "stability_thresholds": list(STABILITY_THRESHOLD_SEQUENCE),
            "stability_mask_applied": stability_flag,
        }

        metadata.update(composite_metadata)
        metadata.update(local_metadata)
        metadata["downloaded_mean_ndvi"] = str(ndvi_path)
        metadata["mean_ndvi_export_task"] = _task_payload(mean_export.task)

        area_stats = (
            local_metadata.get("zones") if isinstance(local_metadata, Mapping) else None
        )
        area_list = area_stats if isinstance(area_stats, list) else []
        if diag_guard is not None:
            diag_guard.record(
                "cartography",
                open_m=open_radius_m,
                close_m=close_radius_m,
                mmu_ha=mmu_value,
                area_stats=area_stats,
            )
            logger.info(
                "zones:cartography open=%s close=%s mmu_ha=%s polys=%s",
                open_radius_m,
                close_radius_m,
                mmu_value,
                len(area_list),
            )
            diag_guard.require(
                bool(area_list),
                "E_VECT_EMPTY",
                "Vectorization produced no polygons.",
                "Loosen MMU; reduce smoothing.",
                area_stats=area_stats,
            )

        return artifacts, metadata

    # Multi-index K-means branch
    composite_images = [image for _, image in composites]
    feature_images_meta = composite_metadata.get("feature_images")
    cleanup_result: CleanupResult
    if isinstance(feature_images_meta, Mapping) and feature_images_meta:
        zone_image, feature_payload, cleanup_result = (
            _build_multiindex_zones_with_features(
                ndvi_stats=ndvi_stats,
                feature_images=feature_images_meta,
                geometry=geometry,
                n_classes=n_classes,
                smooth_radius_m=smooth_radius_m,
                open_radius_m=open_radius_m,
                close_radius_m=close_radius_m,
                min_mapping_unit_ha=mmu_value,
                sample_size=sample_size,
            )
        )
    else:
        zone_image, feature_payload, cleanup_result = _build_multiindex_zones(
            ndvi_stats=ndvi_stats,
            composites=composite_images,
            geometry=geometry,
            n_classes=n_classes,
            smooth_radius_m=smooth_radius_m,
            open_radius_m=open_radius_m,
            close_radius_m=close_radius_m,
            min_mapping_unit_ha=mmu_value,
            sample_size=sample_size,
        )

    if diag_guard is not None:
        diag_guard.record("classification", method=method, breaks=None)
        logger.info("zones:classification method=%s", method)
        logger.info("zones:classification kmeans seed=42")

    zone_raster_path = workdir / "zones_classified.tif"
    zone_export = _download_image_to_path(
        zone_image.rename("zone"), geometry, zone_raster_path
    )
    zone_raster_path = zone_export.path

    with rasterio.open(zone_raster_path) as zone_src:
        classified_data = zone_src.read(1)
        zone_transform = zone_src.transform
        zone_crs = zone_src.crs

    with rasterio.open(ndvi_path) as ndvi_src:
        ndvi_values = ndvi_src.read(1, masked=True)

    smoothing_requested = {
        "smooth_radius_m": float(max(smooth_radius_m, 0)),
        "open_radius_m": float(max(open_radius_m, 0)),
        "close_radius_m": float(max(close_radius_m, 0)),
        "min_mapping_unit_ha": float(max(mmu_value, 0)),
    }
    applied_operations = dict(cleanup_result.applied_operations)
    executed_operations = dict(cleanup_result.executed_operations)
    mmu_was_applied = bool(applied_operations.get("min_mapping_unit"))

    populated = classified_data[classified_data > 0]
    final_zone_count = int(np.unique(populated).size) if populated.size else 0
    effective_zone_count = min(n_classes, final_zone_count) if final_zone_count else 0

    artifacts, local_metadata = _assemble_zone_artifacts(
        classified=classified_data,
        ndvi=ndvi_values,
        transform=zone_transform,
        crs=zone_crs,
        working_dir=workdir,
        include_stats=include_stats,
        raster_path=zone_raster_path,
        mean_ndvi_path=ndvi_path,
        smoothing_requested=smoothing_requested,
        applied_operations=applied_operations,
        executed_operations=executed_operations,
        fallback_applied=bool(cleanup_result.fallback_applied),
        fallback_removed=list(cleanup_result.fallback_removed),
        min_mapping_unit_ha=mmu_value,
        requested_zone_count=n_classes,
        effective_zone_count=effective_zone_count,
        classification_method="multiindex_kmeans",
        thresholds=[],
        kmeans_fallback_applied=False,
        kmeans_cluster_centers=[],
    )

    feature_names = sorted(str(name) for name in feature_payload.keys())

    metadata = {
        "used_months": ordered_months,
        "skipped_months": skipped_months,
        "min_mapping_unit_applied": mmu_was_applied,
        "mmu_applied": mmu_was_applied,
        "zone_method": method,
        "multiindex_feature_names": feature_names,
        "multiindex_feature_count": len(feature_payload),
        "multiindex_sample_size": int(sample_size),
        "stability_thresholds": list(STABILITY_THRESHOLD_SEQUENCE),
        "stability_mask_applied": stability_flag,
    }

    metadata.update(composite_metadata)
    metadata.update(local_metadata)
    metadata["downloaded_mean_ndvi"] = str(ndvi_path)
    metadata["downloaded_zone_raster"] = str(zone_raster_path)
    metadata["mean_ndvi_export_task"] = _task_payload(mean_export.task)
    metadata["zone_raster_export_task"] = _task_payload(zone_export.task)

    area_stats = (
        local_metadata.get("zones") if isinstance(local_metadata, Mapping) else None
    )
    area_list = area_stats if isinstance(area_stats, list) else []
    if diag_guard is not None:
        diag_guard.record(
            "cartography",
            open_m=open_radius_m,
            close_m=close_radius_m,
            mmu_ha=mmu_value,
            area_stats=area_stats,
        )
        logger.info(
            "zones:cartography open=%s close=%s mmu_ha=%s polys=%s",
            open_radius_m,
            close_radius_m,
            mmu_value,
            len(area_list),
        )
        diag_guard.require(
            bool(area_list),
            "E_VECT_EMPTY",
            "Vectorization produced no polygons.",
            "Loosen MMU; reduce smoothing.",
            area_stats=area_stats,
        )

    return artifacts, metadata


def build_zone_artifacts(
    aoi_geojson: dict | ee.Geometry,
    *,
    months: Sequence[str],
    cloud_prob_max: int = DEFAULT_CLOUD_PROB_MAX,
    n_classes: int = DEFAULT_N_CLASSES,
    cv_mask_threshold: float = DEFAULT_CV_THRESHOLD,
    min_obs_for_cv: int = 3,
    apply_stability_mask: bool | None = None,
    min_valid_ratio: float = 0.25,
    min_mapping_unit_ha: float = DEFAULT_MIN_MAPPING_UNIT_HA,
    smooth_radius_m: float = DEFAULT_SMOOTH_RADIUS_M,
    open_radius_m: float = DEFAULT_OPEN_RADIUS_M,
    close_radius_m: float = DEFAULT_CLOSE_RADIUS_M,
    simplify_tolerance_m: float = DEFAULT_SIMPLIFY_TOL_M,
    simplify_buffer_m: float = DEFAULT_SIMPLIFY_BUFFER_M,
    method: str = DEFAULT_METHOD,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    include_stats: bool = True,
) -> ZoneArtifacts:
    if n_classes < 3 or n_classes > 7:
        raise ValueError("n_classes must be between 3 and 7")
    if min_mapping_unit_ha <= 0:
        raise ValueError("min_mapping_unit_ha must be positive")
    if smooth_radius_m < 0 or open_radius_m < 0 or close_radius_m < 0:
        raise ValueError("Smoothing radii must be non-negative")
    if not months:
        raise ValueError("At least one month must be supplied")

    method_key = (method or "").strip().lower()
    if not method_key:
        method_key = DEFAULT_METHOD
    if method_key not in {"ndvi_percentiles", "multiindex_kmeans", "ndvi_kmeans"}:
        raise ValueError("Unsupported method for production zones")
    try:
        gee.initialize()
    except Exception:  # pragma: no cover - test fallback
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
        min_obs_for_cv=min_obs_for_cv,
        apply_stability_mask=apply_stability_mask,
        min_valid_ratio=min_valid_ratio,
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


def start_zone_exports(
    artifacts: ZoneArtifacts,
    *,
    aoi_name: str,
    months: Sequence[str] | None,
    bucket: str,
    include_stats: bool = True,
    prefix_override: str | None = None,
) -> dict[str, ee.batch.Task]:
    raise RuntimeError(
        "Cloud exports are not supported for locally generated zone artifacts"
    )


def start_zone_exports_drive(
    artifacts: ZoneArtifacts,
    *,
    folder: str,
    prefix: str,
    include_stats: bool = True,
) -> dict[str, ee.batch.Task]:
    raise RuntimeError(
        "Drive exports are not supported for locally generated zone artifacts"
    )


def _task_payload(task: ee.batch.Task | None) -> dict[str, object]:
    if task is None:
        return {}
    payload: dict[str, object] = {"id": getattr(task, "id", None)}
    try:
        status = task.status() or {}
    except Exception:  # pragma: no cover
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


def export_selected_period_zones(
    aoi_geojson: dict,
    aoi_name: str,
    months: list[str],
    *,
    geometry: ee.Geometry | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    cloud_prob_max: int = 40,
    mask_mode: str = "adaptive",
    n_classes: int = 5,
    cv_mask_threshold: float = DEFAULT_CV_THRESHOLD,
    min_obs_for_cv: int = 3,
    stability_adaptive: bool = True,
    stability_enforce: bool = False,
    mmu_ha: float = 2.0,
    min_mapping_unit_ha: float | None = None,
    smooth_radius_m: int = 30,
    open_radius_m: int = 10,
    close_radius_m: int = 10,
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
    min_valid_ratio: float = 0.25,
    method: str | None = None,
    diagnostics: bool = False,
    debug_dump: bool = False,
):
    g = Guard()
    logger.info(
        "zones:start aoi=%s n_classes=%s diagnostics=%s",
        aoi_name or "NA",
        n_classes,
        diagnostics,
    )
    working_dir = _ensure_working_directory(None)

    aoi = _to_ee_geometry(aoi_geojson)
    geometry = geometry or aoi
    area_m2: float | None = None
    try:
        area_value = geometry.area(maxError=1)
        if hasattr(area_value, "getInfo"):
            area_m2 = float(ee.Number(area_value).getInfo() or 0)
        else:
            area_m2 = float(area_value)
    except Exception:
        try:
            if isinstance(aoi_geojson, dict):
                area_m2 = float(area_ha(aoi_geojson) * 10_000)
        except Exception:
            area_m2 = None
    g.record(
        "inputs",
        area_m2=area_m2,
        start=str(start_date) if start_date is not None else None,
        end=str(end_date) if end_date is not None else None,
        n_classes=n_classes,
    )
    logger.info(
        "zones:inputs area_m2=%.2f start=%s end=%s n=%s",
        area_m2 if area_m2 is not None else -1,
        start_date,
        end_date,
        n_classes,
    )
    g.require(
        bool(area_m2) and area_m2 > 1000,
        "E_INPUT_AOI",
        "AOI is empty or too small.",
        "Fix geometry/CRS; buffer.",
        area_m2=area_m2,
    )
    if start_date is not None and end_date is not None and end_date < start_date:
        raise ValueError("end_date must be on or after start_date")

    if min_mapping_unit_ha is not None:
        mmu_ha = float(min_mapping_unit_ha)
    if simplify_tolerance_m is not None:
        simplify_tol_m = int(simplify_tolerance_m)
    if destination is not None:
        export_target = destination

    if not months:
        if start_date is None or end_date is None:
            raise ValueError("Either months or start/end dates must be supplied")
        months = _months_from_dates(start_date, end_date)

    ordered_months = _ordered_months(months)
    if start_date is None or end_date is None:
        start_date, end_date = _month_range_dates(ordered_months)

    include_stats_flag = bool(
        include_stats if include_stats is not None else include_zonal_stats
    )

    try:
        gee.initialize()
    except Exception:  # pragma: no cover
        if not _allow_init_failure():
            raise
    geometry = geometry or _resolve_geometry(aoi_geojson)

    method_selection = (method or "").strip().lower()
    if not method_selection:
        method_selection = DEFAULT_METHOD
    if method_selection not in {"ndvi_percentiles", "multiindex_kmeans", "ndvi_kmeans"}:
        raise ValueError("Unsupported method for production zones")

    try:
        artifacts, metadata = _prepare_selected_period_artifacts(
            aoi_geojson,
            geometry=geometry,
            working_dir=working_dir,
            months=ordered_months,
            start_date=start_date,
            end_date=end_date,
            cloud_prob_max=cloud_prob_max,
            mask_mode=mask_mode,
            n_classes=n_classes,
            cv_mask_threshold=cv_mask_threshold,
            min_obs_for_cv=min_obs_for_cv,
            apply_stability_mask=apply_stability_mask,
            stability_adaptive=stability_adaptive,
            stability_enforce=stability_enforce,
            min_valid_ratio=min_valid_ratio,
            min_mapping_unit_ha=mmu_ha,
            smooth_radius_m=smooth_radius_m,
            open_radius_m=open_radius_m,
            close_radius_m=close_radius_m,
            simplify_tol_m=simplify_tol_m,
            simplify_buffer_m=simplify_buffer_m,
            method=method_selection,
            sample_size=DEFAULT_SAMPLE_SIZE,
            include_stats=include_stats_flag,
            debug_dump=debug_dump,
            guard=g,
        )
    except PipelineError:
        raise

    metadata = dict(metadata)
    metadata["zone_method"] = method_selection
    metadata = sanitize_for_json(metadata)
    used_months: list[str] = list(metadata.get("used_months", []))
    skipped: list[str] = list(metadata.get("skipped_months", []))
    if not used_months:
        raise ValueError("No valid Sentinel-2 scenes available for the selected period")

    prefix_base = export_prefix(aoi_name, used_months)

    metadata.update(
        {
            "used_months": used_months,
            "skipped_months": skipped,
        }
    )

    result: dict[str, object] = {
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

    palette = metadata.get("palette") if isinstance(metadata, dict) else None
    thresholds = (
        metadata.get("percentile_thresholds") if isinstance(metadata, dict) else None
    )
    if palette is not None:
        result["palette"] = palette
    if thresholds is not None:
        result["thresholds"] = thresholds

    export_target = (export_target or "zip").strip().lower()
    if export_target not in {"zip", "local"}:
        raise ValueError("Only local zone exports are supported in this workflow")

    diag = g.diagnostics_payload()
    if debug_dump:
        try:
            diag_bytes = json.dumps(diag, ensure_ascii=False, indent=2).encode("utf-8")
            extra_files = dict(artifacts.extra_files or {})
            extra_files["diagnostics/diagnostics.json"] = diag_bytes
            artifacts = replace(artifacts, extra_files=extra_files)
            result["artifacts"] = artifacts
        except Exception as _exc:
            logger.warning("debug_dump: failed to attach diagnostics.json: %s", _exc)
    if diagnostics:
        result = dict(result)
        result["ok"] = True
        result["diagnostics"] = diag
        try:
            logger.debug("zones:diagnostics payload=%s", json.dumps(diag))
        except Exception:  # pragma: no cover - diagnostics logging guard
            logger.debug("zones:diagnostics payload serialization failed")

    return result
