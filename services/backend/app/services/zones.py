"""Production zone workflow built on Sentinel-2 monthly composites."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import calendar
import csv
import io
import logging
import os
import math
from pathlib import Path
import re
import shutil
import tempfile
import zipfile
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union
from urllib.request import urlopen

import ee
import numpy as np
import rasterio
from rasterio.features import shapes
import shapefile
from pyproj import CRS
from shapely.geometry import mapping, shape
from sklearn.cluster import KMeans
from scipy import ndimage

from app import gee
from app.exports import sanitize_name
from app.services.image_stats import temporal_stats
from app.utils.geometry import area_ha
from app.utils.sanitization import sanitize_for_json


logger = logging.getLogger(__name__)


DEFAULT_CLOUD_PROB_MAX = 40
DEFAULT_N_CLASSES = 5
DEFAULT_CV_THRESHOLD = 0.5
DEFAULT_MIN_MAPPING_UNIT_HA = 1.5
DEFAULT_SMOOTH_RADIUS_M = 30
DEFAULT_OPEN_RADIUS_M = 10
DEFAULT_CLOSE_RADIUS_M = 10
DEFAULT_SIMPLIFY_TOL_M = 5
DEFAULT_SIMPLIFY_BUFFER_M = 3
DEFAULT_METHOD = "ndvi_percentiles"
DEFAULT_SAMPLE_SIZE = 8000
DEFAULT_SCALE = 10
# IMPORTANT: processing uses the native S2 projection (meters).
# Exports use a metric CRS so scale=10 means 10 meters.
DEFAULT_EXPORT_CRS = "EPSG:3857"
DEFAULT_CRS = DEFAULT_EXPORT_CRS


def _allow_init_failure() -> bool:
    flag = os.getenv("GEE_ALLOW_INIT_FAILURE", "")
    return flag.strip().lower() in {"1", "true", "yes"}


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


def _geometry_region(geometry: ee.Geometry) -> List[List[List[float]]]:
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
    return payload.startswith(b"PK\x03\x04") or payload.startswith(b"PK\x05\x06") or payload.startswith(
        b"PK\x07\x08"
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
) -> tuple[ZoneArtifacts, Dict[str, object]]:
    pixel_size_x = abs(transform.a)
    pixel_size_y = abs(transform.e)
    pixel_area = pixel_size_x * pixel_size_y

    classified_array = np.asarray(classified)
    if classified_array.dtype != np.uint8:
        classified_array = classified_array.astype(np.uint8)

    unique_zones = np.unique(classified_array[classified_array > 0])

    records: List[Dict[str, object]] = []
    zonal_stats: List[Dict[str, object]] = []
    for geom, value in shapes(classified_array, mask=classified_array > 0, transform=transform):
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
                fieldnames=["zone", "area_ha", "mean_ndvi", "min_ndvi", "max_ndvi", "pixel_count"],
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
        stage_key_map[name]: bool(executed_operations.get(name)) for name in stage_key_map
    }

    skipped_steps = [stage_key_map[name] for name in fallback_removed if name in stage_key_map]
    skipped_steps = sorted(set(skipped_steps))

    final_zone_count = int(unique_zones.size)
    palette: list[str] = list(
        ZONE_PALETTE[: max(1, min(final_zone_count, len(ZONE_PALETTE)))]
    )

    metadata: Dict[str, object] = {
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
) -> tuple[ZoneArtifacts, Dict[str, object]]:
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
    raw_thresholds = (
        np.nanpercentile(valid_values, percentiles) if percentiles.size else np.array([])
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
                        found_idx = int(np.searchsorted(boundary_props, target, side="left"))
                        if found_idx >= B:
                            found_idx = B - 1
                        idx = min(max(found_idx, min_idx), max_idx)
                        selected_indices.append(idx)
                        prev_idx = idx
                    fallback_thresholds = np.array(
                        [
                            (float(unique_sorted[idx]) + float(unique_sorted[idx + 1])) / 2.0
                            for idx in selected_indices
                        ],
                        dtype=float,
                    )
                    fallback_comparison = comparison_data
                    fallback_classified = (
                        np.sum(fallback_comparison > fallback_thresholds, axis=-1, dtype=np.int16) + 1
                    )
                    fallback_classified[combined_mask] = 0
                    fallback_unique = np.unique(fallback_classified[fallback_classified > 0])
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
    unique_zones = np.unique(stage_results[final_index][1][stage_results[final_index][1] > 0])
    while unique_zones.size < effective_n_classes and final_index > 0:
        stage_name = stage_results[final_index][0]
        if stage_name in applied_operations and applied_operations[stage_name]:
            applied_operations[stage_name] = False
            fallback_removed.append(stage_name)
        final_index -= 1
        unique_zones = np.unique(stage_results[final_index][1][stage_results[final_index][1] > 0])

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


def _ordered_months(months: Sequence[str]) -> List[str]:
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


def _months_from_dates(start_date: date, end_date: date) -> List[str]:
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")
    months: List[str] = []
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
    return f"zones/PROD_{start.replace('-', '')}_{end.replace('-', '')}_{safe_name}_zones"


def export_prefix(aoi_name: str, months: Sequence[str]) -> str:
    """Public helper that returns the export prefix for a zone export run."""
    return _export_prefix(aoi_name, months)


def month_bounds(months: Sequence[str]) -> tuple[str, str]:
    """Return the first and last months used in an export request."""
    return _month_bounds(months)


def resolve_export_bucket(explicit: str | None = None) -> str:
    bucket = (explicit or os.getenv("GEE_GCS_BUCKET") or os.getenv("GCS_BUCKET") or "").strip()
    if not bucket:
        raise RuntimeError("GEE_GCS_BUCKET or GCS_BUCKET must be set for zone exports")
    return bucket


def _resolve_geometry(aoi: Union[dict, ee.Geometry]) -> ee.Geometry:
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
    qa_mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))

    prob_mask = image.select("cloud_probability").lte(cloud_prob_max)

    scl = image.select("SCL")
    shadow_mask = scl.neq(3).And(scl.neq(11))

    combined_mask = qa_mask.And(prob_mask).And(shadow_mask)
    scaled = image.updateMask(combined_mask).divide(10_000)
    selected = scaled.select(list(gee.S2_BANDS))
    return selected.copyProperties(image, ["system:time_start"])


def _native_reproject(img: ee.Image) -> ee.Image:
    """Reproject to native Sentinel-2 10 m projection (meters)."""
    # Use B8 (10 m) projection as the reference
    proj = img.select("B8").projection()
    return img.resample("bilinear").reproject(proj, None, DEFAULT_SCALE)


def _build_masked_s2_collection(*_args, **_kwargs):  # pragma: no cover - compatibility shim
    raise NotImplementedError("_build_masked_s2_collection is not implemented in this module")


def _build_composite_collection(*_args, **_kwargs):  # pragma: no cover - compatibility shim
    raise NotImplementedError("_build_composite_collection is not implemented in this module")


def _ndvi_collection(*_args, **_kwargs):  # pragma: no cover - compatibility shim
    raise NotImplementedError("_ndvi_collection is not implemented in this module")


def _ndvi_statistics(*_args, **_kwargs):  # pragma: no cover - compatibility shim
    raise NotImplementedError("_ndvi_statistics is not implemented in this module")


def _ndvi_percentile_thresholds(*_args, **_kwargs):  # pragma: no cover - compatibility shim
    raise NotImplementedError("_ndvi_percentile_thresholds is not implemented in this module")


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
) -> Tuple[List[tuple[str, ee.Image]], List[str], Dict[str, object]]:
    composites: List[tuple[str, ee.Image]] = []
    skipped: List[str] = []
    metadata: Dict[str, object] = {}
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

            # ⬇⬇ FIX: use native S2 projection (meters), not EPSG:4326
            reproj = _native_reproject(composite)
            ndvi = _compute_ndvi(reproj)
            try:
                valid_pixels = int(
                    ee.Number(
                        ndvi.mask()
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

            composites.append((month, reproj.clip(geometry)))
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


def _compute_ndvi(image: ee.Image) -> ee.Image:
    bands = image.select(["B8", "B4"]).toFloat()
    ndvi = bands.normalizedDifference(["B8", "B4"]).rename("NDVI")
    return ndvi.updateMask(image.mask())


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


def _ndvi_temporal_stats(images: Sequence[ee.Image]) -> Mapping[str, ee.Image]:
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

    threshold_list = ee.List([float(t) for t in thresholds])
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
) -> List[float]:
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
    cut_lookup: Dict[int, float] = {}
    percentile_lookup: List[tuple[float, float]] = []

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

    thresholds: List[float] = []
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
) -> tuple[ee.Image, List[float]]:
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
    pct_breaks = ee.List(percentile_sequence)
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
    thresholds: List[float]
    try:
        thresholds = _percentile_thresholds(
            reducer_info, percentile_sequence, band_label
        )
    except ValueError as exc:
        raise ValueError(STABILITY_MASK_EMPTY_ERROR) from exc

    adjusted_thresholds: List[float] = []
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

    # Now classify pixels relative to thresholds
    zero = image.multiply(0)

    def _accumulate(current, threshold):
        current_img = ee.Image(current)
        t = ee.Number(threshold)
        gt_band = image.gt(t)
        return current_img.add(gt_band)

    summed = ee.List(thresholds).iterate(_accumulate, zero)
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
        smoothed = classified.focal_mode(radius=smooth_radius, units="meters", iterations=1)

    opened = smoothed
    if open_radius > 0:
        opened = (
            smoothed.focal_min(radius=open_radius, units="meters", iterations=1)
            .focal_max(radius=open_radius, units="meters", iterations=1)
        )

    closed = opened
    if close_radius > 0:
        closed = (
            opened.focal_max(radius=close_radius, units="meters", iterations=1)
            .focal_min(radius=close_radius, units="meters", iterations=1)
        )

    component_area = _connected_component_area(closed, n_classes)
    min_area_m2 = max(min_mapping_unit_ha, 0) * 10_000
    majority_large = closed
    if smooth_radius > 0:
        majority_large = closed.focal_mode(radius=smooth_radius, units="meters", iterations=1)
    small_mask = component_area.lt(min_area_m2)
    if hasattr(small_mask, "rename") and hasattr(classified, "bandNames"):
        small_mask = small_mask.rename(classified.bandNames())
    cleaned = closed.where(small_mask, majority_large)

    mask = cleaned.mask()
    closed_mask = mask
    if close_radius > 0:
        closed_mask = mask.focal_max(radius=close_radius, units="meters", iterations=1).focal_min(
            radius=close_radius, units="meters", iterations=1
        )
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
    applied_operations: Dict[str, bool]
    executed_operations: Dict[str, bool]
    fallback_applied: bool
    fallback_removed: List[str]


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

    applied_operations: Dict[str, bool] = {
        "smooth": bool(smooth_radius_m > 0),
        "open": bool(open_radius_m > 0),
        "close": bool(close_radius_m > 0),
        "min_mapping_unit": bool(mmu_value > 0),
    }
    executed_operations = dict(applied_operations)

    stage_names: List[str] = []
    stage_images: List[ee.Image] = []
    stage_counts: List[int] = []

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

    fallback_removed: List[str] = []
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
        return (
            feature.setGeometry(geom)
            .set({
                "zone": zone_value,
                "zone_id": zone_value,
                "area_m2": area_m2,
                "area_ha": area_ha_val,
            })
        )

    return ee.FeatureCollection(vectors.map(_simplify))


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

    return _simplify_vectors(vectors, tolerance_m, buffer_m)


def _collect_stats_images(
    ndvi_stats: Mapping[str, ee.Image],
    extra_means: Mapping[str, ee.Image] | None = None,
) -> Dict[str, ee.Image]:
    stats: Dict[str, ee.Image] = {
        "NDVI_mean": ndvi_stats["mean"],
        "NDVI_median": ndvi_stats["median"],
        "NDVI_stdDev": ndvi_stats["std"],
        "NDVI_cv": ndvi_stats["cv"],
    }
    if extra_means:
        for key, image in extra_means.items():
            stats[key] = image
    return stats


def _add_zonal_stats(feature: ee.Feature, stats_images: Mapping[str, ee.Image]) -> ee.Feature:
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

    keys = ee.List(stats_dict.keys())

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

    sanitized_stats = ee.Dictionary(keys.map(_sanitize_value).iterate(_merge, ee.Dictionary({})))
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
) -> tuple[ee.Image, List[float]]:
    # Cap NDVI for percentile breaks only (0..0.6)
    pct_source = ndvi_stats["mean"].clamp(NDVI_PERCENTILE_MIN, NDVI_PERCENTILE_MAX)

    # Robust thresholds: compute on mean mask (not stability) to avoid empty stats
    thresholds_image = pct_source.updateMask(ndvi_stats["mean"].mask())
    ranked_for_thresh, thresholds = _classify_by_percentiles(
        thresholds_image, geometry, n_classes
    )
    # Now classify the full pct_source, then apply stability mask afterwards
    ranked_full, _ = _classify_by_percentiles(
        pct_source, geometry, n_classes
    )
    ranked = ranked_full.updateMask(ndvi_stats["stability"])

    try:
        percentile_thresholds = [float(value) for value in thresholds]
    except (TypeError, ValueError) as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to evaluate NDVI percentile thresholds: {exc}") from exc

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


def _normalise_feature(mean_image: ee.Image, geometry: ee.Geometry, name: str) -> ee.Image:
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
        except (TypeError, KeyError, IndexError, AttributeError):  # pragma: no cover - defensive
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


def _rank_zones(cluster_image: ee.Image, ndvi_mean: ee.Image, geometry: ee.Geometry) -> ee.Image:
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
    groups = ee.List(grouped.get("groups", ee.List([])))

    def _cluster_value(item):
        info = ee.Dictionary(item)
        return ee.Dictionary(
            {
                "cluster": ee.Number(info.get("cluster")),
                "mean_ndvi": ee.Number(info.get("mean_ndvi", 0)),
            }
        )

    sorted_groups = groups.map(_cluster_value).sort("mean_ndvi")
    source = ee.List(sorted_groups.map(lambda g: ee.Number(ee.Dictionary(g).get("cluster"))))
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
    ndvi_images: Sequence[ee.Image],
    ndvi_stats: Mapping[str, ee.Image],
) -> Dict[str, ee.Image]:
    collection = ee.ImageCollection([image.rename("NDVI") for image in ndvi_images])
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
    ndvi_images: Sequence[ee.Image],
    ndvi_stats: Mapping[str, ee.Image],
    geometry: ee.Geometry,
    n_classes: int,
    smooth_radius_m: float,
    open_radius_m: float,
    close_radius_m: float,
    min_mapping_unit_ha: float,
    sample_size: int,
    rank_by_mean: bool = True,
) -> tuple[ee.Image, Dict[str, ee.Image], CleanupResult]:
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
) -> tuple[ee.Image, Dict[str, ee.Image], CleanupResult]:
    indices = {
        "NDVI": [image.normalizedDifference(["B8", "B4"]).rename("NDVI") for image in composites],
        "NDRE": [
            image.normalizedDifference(["B8", "B5"]).rename("NDRE") for image in composites
        ],
        "NDMI": [
            image.normalizedDifference(["B8", "B11"]).rename("NDMI") for image in composites
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

    mean_images: Dict[str, ee.Image] = {}
    for name, stack in indices.items():
        collection = ee.ImageCollection(stack)
        mean_images[name] = collection.mean().rename(f"{name}_mean")

    normalised = [_normalise_feature(mean_images[name], geometry, name) for name in sorted(mean_images)]
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
) -> tuple[ee.Image, Dict[str, ee.Image], CleanupResult]:
    stability = ndvi_stats["stability"]
    masked_features: Dict[str, ee.Image] = {}
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
    _ = (
        cv_mask_threshold,
        apply_stability_mask,
        simplify_tol_m,
        simplify_buffer_m,
        method,
        sample_size,
    )
    ordered_months = _ordered_months(months)
    composites, skipped_months, composite_metadata = _build_composite_series(
        geometry,
        ordered_months,
        start_date,
        end_date,
        cloud_prob_max,
    )
    if not composites:
        raise ValueError("No valid Sentinel-2 scenes were found for the selected months")

    ndvi_images = [_compute_ndvi(image) for _, image in composites]
    ndvi_stats = dict(_ndvi_temporal_stats(ndvi_images))

    diag_region = geometry
    try:
        collection = ee.ImageCollection(ndvi_images)
        size_value = int(ee.Number(collection.size()).getInfo() or 0)
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

    stability_flag = APPLY_STABILITY if apply_stability_mask is None else bool(apply_stability_mask)
    if stability_flag:
        stability_image = _stability_mask(
            ndvi_stats["cv"],
            geometry,
            STABILITY_THRESHOLD_SEQUENCE,
            MIN_STABILITY_SURVIVAL_RATIO,
            DEFAULT_SCALE,
        )
    else:
        stability_image = ee.Image(1)
    ndvi_stats["stability"] = stability_image

    mean_image = ndvi_stats["mean"].rename("NDVI_mean")

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

    if method == "ndvi_percentiles":
        artifacts, local_metadata = _classify_local_zones(
            ndvi_path,
            working_dir=workdir,
            n_classes=n_classes,
            min_mapping_unit_ha=mmu_value,
            smooth_radius_m=smooth_radius_m,
            open_radius_m=open_radius_m,
            close_radius_m=close_radius_m,
            include_stats=include_stats,
        )

        mmu_was_applied = mmu_value > 0 and mmu_applied
        metadata: Dict[str, object] = {
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

        return artifacts, metadata

    if method == "ndvi_kmeans":
        zone_image, feature_payload, cleanup_result = _build_ndvi_kmeans_zones(
            ndvi_images=ndvi_images,
            ndvi_stats=ndvi_stats,
            geometry=geometry,
            n_classes=n_classes,
            smooth_radius_m=smooth_radius_m,
            open_radius_m=open_radius_m,
            close_radius_m=close_radius_m,
            min_mapping_unit_ha=mmu_value,
            sample_size=sample_size,
        )

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

        return artifacts, metadata

    # Multi-index K-means branch
    composite_images = [image for _, image in composites]
    feature_images_meta = composite_metadata.get("feature_images")
    cleanup_result: CleanupResult
    if isinstance(feature_images_meta, Mapping) and feature_images_meta:
        zone_image, feature_payload, cleanup_result = _build_multiindex_zones_with_features(
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

    zone_raster_path = workdir / "zones_classified.tif"
    zone_export = _download_image_to_path(zone_image.rename("zone"), geometry, zone_raster_path)
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

    return artifacts, metadata


def build_zone_artifacts(
    aoi_geojson: Union[dict, ee.Geometry],
    *,
    months: Sequence[str],
    cloud_prob_max: int = DEFAULT_CLOUD_PROB_MAX,
    n_classes: int = DEFAULT_N_CLASSES,
    cv_mask_threshold: float = DEFAULT_CV_THRESHOLD,
    apply_stability_mask: bool | None = None,
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


def start_zone_exports(
    artifacts: ZoneArtifacts,
    *,
    aoi_name: str,
    months: Sequence[str] | None,
    bucket: str,
    include_stats: bool = True,
    prefix_override: str | None = None,
) -> Dict[str, ee.batch.Task]:
    raise RuntimeError(
        "Cloud exports are not supported for locally generated zone artifacts"
    )


def start_zone_exports_drive(
    artifacts: ZoneArtifacts,
    *,
    folder: str,
    prefix: str,
    include_stats: bool = True,
) -> Dict[str, ee.batch.Task]:
    raise RuntimeError(
        "Drive exports are not supported for locally generated zone artifacts"
    )


def _task_payload(task: ee.batch.Task | None) -> Dict[str, object]:
    if task is None:
        return {}
    payload: Dict[str, object] = {"id": getattr(task, "id", None)}
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
    n_classes: int = 5,
    cv_mask_threshold: float | None = None,
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
    method: str | None = None,
):
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

    artifacts, metadata = _prepare_selected_period_artifacts(
        aoi_geojson,
        geometry=geometry,
        working_dir=working_dir,
        months=ordered_months,
        start_date=start_date,
        end_date=end_date,
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
        method=method_selection,
        sample_size=DEFAULT_SAMPLE_SIZE,
        include_stats=include_stats_flag,
    )

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

    result: Dict[str, object] = {
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
    thresholds = metadata.get("percentile_thresholds") if isinstance(metadata, dict) else None
    if palette is not None:
        result["palette"] = palette
    if thresholds is not None:
        result["thresholds"] = thresholds

    export_target = (export_target or "zip").strip().lower()
    if export_target not in {"zip", "local"}:
        raise ValueError("Only local zone exports are supported in this workflow")

    return result
