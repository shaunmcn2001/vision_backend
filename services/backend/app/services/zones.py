# services/backend/app/services/zones.py
# Patched: mean NDVI computation (sum/count), explicit export CRS, 10 m native processing,
# streaming NDVI percentiles + classification to avoid OOM, Drive export fix.

from __future__ import annotations

import calendar
import csv
import io
import logging
import math
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union
from urllib.request import urlopen
from zipfile import ZipFile

import ee
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.windows import Window
import shapefile
from shapely.geometry import mapping, shape
from pyproj import CRS

from app import gee
from app.exports import sanitize_name
from app.services.image_stats import temporal_stats
from app.services.ndvi_shared import (
    compute_ndvi_loose,
    mean_from_collection_sum_count,
    reproject_native_10m,
)
from app.utils.geometry import area_ha
from app.utils.sanitization import sanitize_for_json

logger = logging.getLogger(__name__)

DEFAULT_CLOUD_PROB_MAX = 40
DEFAULT_N_CLASSES = 5
DEFAULT_CV_THRESHOLD = 0.5
DEFAULT_MIN_MAPPING_UNIT_HA = 1.5

# Disable smoothing in this OOM-safe build (classification happens streaming).
DEFAULT_SMOOTH_RADIUS_M = 0
DEFAULT_OPEN_RADIUS_M = 0
DEFAULT_CLOSE_RADIUS_M = 0
DEFAULT_SIMPLIFY_TOL_M = 5
DEFAULT_SIMPLIFY_BUFFER_M = 3
DEFAULT_METHOD = "ndvi_percentiles"
DEFAULT_SAMPLE_SIZE = 4000

DEFAULT_SCALE = int(os.getenv("ZONES_SCALE_M", "10"))
DEFAULT_EXPORT_CRS = "EPSG:3857"  # GeoTIFF CRS
DEFAULT_CRS = DEFAULT_EXPORT_CRS

ZONES_STREAM_TRIGGER_PX = int(os.getenv("ZONES_STREAM_TRIGGER_PX", "40000000"))  # ~40M px

ZONE_PALETTE: tuple[str, ...] = (
    "#112f1d", "#1b4d2a", "#2c6a39", "#3f8749", "#58a35d", "#80bf7d", "#b6dcb1",
)

NDVI_PERCENTILE_MIN = 0.0
NDVI_PERCENTILE_MAX = 0.6

NDVI_MASK_EMPTY_ERROR = (
    "No valid NDVI pixels across the selected months. Try a different date range or AOI."
)

def _parse_bool_env(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    t = value.strip().lower()
    if t in {'', 'none'}: return default
    if t in {'0', 'false', 'no', 'off'}: return False
    if t in {'1', 'true', 'yes', 'on'}: return True
    return default

APPLY_STABILITY = _parse_bool_env(os.getenv('APPLY_STABILITY'), True)

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
    flag = os.getenv('GEE_ALLOW_INIT_FAILURE', '')
    return flag.strip().lower() in {'1', 'true', 'yes'}

def _to_ee_geometry(geojson: dict) -> ee.Geometry:
    try:
        t = (geojson or {}).get('type', '')
        if t == 'Feature':
            return ee.Feature(geojson).geometry()
        if t == 'FeatureCollection':
            return ee.FeatureCollection(geojson).geometry()
        return ee.Geometry(geojson)
    except Exception as e:
        raise ValueError(f'Invalid AOI GeoJSON: {e}')

def _ensure_working_directory(path: os.PathLike[str] | str | None) -> Path:
    if path:
        p = Path(path); p.mkdir(parents=True, exist_ok=True); return p
    return Path(tempfile.mkdtemp(prefix='zones_'))

def _as_lists(value: Any) -> Any:
    if isinstance(value, tuple): return [_as_lists(v) for v in value]
    if isinstance(value, list): return [_as_lists(v) for v in value]
    return value

def _geometry_region(geometry: ee.Geometry) -> List[List[List[float]]]:
    info = geometry.getInfo()
    if not info: raise ValueError('Unable to resolve geometry information for download')
    geom_type = info.get('type')
    if geom_type == 'Polygon':
        coords = info.get('coordinates')
        if coords: return _as_lists(coords)
        raise ValueError('Geometry information missing coordinates')
    if geom_type == 'MultiPolygon':
        dissolved = geometry.dissolve()
        di = dissolved.getInfo() or info
        if di.get('type') == 'Polygon':
            return _as_lists(di.get('coordinates', []))
    coords = info.get('coordinates')
    if coords is not None: return _as_lists(coords)
    raise ValueError('Geometry information missing coordinates')

def _write_zip_geotiff_from_file(zip_path: Path, target: Path) -> None:
    with ZipFile(zip_path, 'r') as archive:
        members = [m for m in archive.namelist() if m.lower().endswith(('.tif', '.tiff'))]
        if not members:
            raise ValueError('Zip archive did not contain a GeoTIFF file')
        first = members[0]
        with archive.open(first) as source, target.open('wb') as out_f:
            shutil.copyfileobj(source, out_f)

def _attach_cloud_probability(collection: ee.ImageCollection, probability: ee.ImageCollection) -> ee.ImageCollection:
    join = ee.Join.saveFirst('cloud_prob')
    matches = join.apply(
        primary=collection,
        secondary=probability,
        condition=ee.Filter.equals(leftField='system:index', rightField='system:index'),
    )
    def _add_probability(image: ee.Image) -> ee.Image:
        cloud_match = image.get('cloud_prob')
        probability_band = ee.Image(
            ee.Algorithms.If(
                ee.Algorithms.IsEqual(cloud_match, None),
                ee.Image.constant(0),
                ee.Image(cloud_match).select('probability'),
            )
        ).rename('cloud_probability')
        return image.addBands(probability_band)
    return ee.ImageCollection(matches).map(_add_probability)

def _mask_s2(image: ee.Image, cloud_prob_max: int) -> ee.Image:
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    qa_mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    prob_mask = image.select('cloud_probability').lte(cloud_prob_max)
    scl = image.select('SCL')
    shadow_mask = scl.neq(3).And(scl.neq(11))
    combined_mask = qa_mask.And(prob_mask).And(shadow_mask)
    scaled = image.updateMask(combined_mask).divide(10_000)
    selected = scaled.select(list(gee.S2_BANDS))
    return selected.copyProperties(image, ['system:time_start'])

def _build_composite_series(geometry: ee.Geometry, months: Sequence[str], start_date: date, end_date: date, cloud_prob_max: int) -> Tuple[List[tuple[str, ee.Image]], List[str], Dict[str, object]]:
    comps: List[tuple[str, ee.Image]] = []
    skipped: List[str] = []
    meta: Dict[str, object] = {}
    ordered = sorted(set(months))
    month_span = len(ordered)

    start_iso = start_date.isoformat()
    end_exclusive_iso = (end_date + timedelta(days=1)).isoformat()

    if month_span >= 3:
        meta['composite_mode'] = 'monthly'
        for month in ordered:
            collection, composite = gee.monthly_sentinel2_collection(geometry, month, cloud_prob_max)
            scene_count = int(ee.Number(collection.size()).getInfo() or 0)
            if scene_count == 0:
                skipped.append(month)
                continue
            clipped = composite.clip(geometry)
            ndvi = compute_ndvi_loose(clipped)
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
            if valid_pixels == 0:
                skipped.append(month)
                continue
            comps.append((month, clipped))
    else:
        meta['composite_mode'] = 'scene'
        base = (ee.ImageCollection(gee.S2_SR_COLLECTION).filterBounds(geometry).filterDate(start_iso, end_exclusive_iso))
        prob = (ee.ImageCollection(gee.S2_CLOUD_PROB_COLLECTION).filterBounds(geometry).filterDate(start_iso, end_exclusive_iso))
        with_prob = _attach_cloud_probability(base, prob)
        masked = with_prob.map(lambda img: _mask_s2(img, cloud_prob_max))
        scene_count = int(ee.Number(masked.size()).getInfo() or 0)
        meta['scene_count'] = scene_count
        if scene_count == 0: return [], ordered, meta
        image_list = masked.toList(scene_count)
        for idx in range(scene_count):
            image = ee.Image(image_list.get(idx))
            label = f'scene_{idx + 1:02d}'
            comps.append((label, image.clip(geometry)))
    return comps, skipped, meta

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
    image = image.toFloat()  # ensure float32

    try:
        region_candidate = geometry.geometry() if hasattr(geometry, "geometry") else geometry
    except Exception:
        region_candidate = geometry
    if isinstance(region_candidate, ee.Geometry):
        ee_region = region_candidate
    else:
        region_coords = _geometry_region(geometry)
        ee_region = ee.Geometry.Polygon(region_coords)
    sanitized_name = sanitize_name(target.stem or 'export')
    description = f'zones_{sanitized_name}'[:100]
    folder = os.getenv('GEE_DRIVE_FOLDER', 'Sentinel2_Zones')

    task: ee.batch.Task | None = None
    try:
        export_kwargs: Dict[str, object] = {
            'image': image,
            'description': description,
            'folder': folder,
            'fileNamePrefix': sanitized_name,
            'region': ee_region,
            'scale': params.scale,
            'fileFormat': 'GeoTIFF',
            'maxPixels': gee.MAX_PIXELS,
        }
        if params.crs:
            export_kwargs['crs'] = params.crs
        task = ee.batch.Export.image.toDrive(**export_kwargs)
        task.start()
    except Exception:
        logger.exception('Failed to start Drive export for %s', sanitized_name)
        task = None

    dl_params: Dict[str, object] = {
        'scale': params.scale,
        'region': ee_region,
        'filePerBand': False,   # only for getDownloadURL
        'format': 'GeoTIFF',
    }
    if params.crs:
        dl_params['crs'] = params.crs
    dl_params['noData'] = -32768
    url = image.getDownloadURL(dl_params)
    with urlopen(url) as response:
        headers = getattr(response, 'headers', None)
        content_type = headers.get('Content-Type', '') if headers and hasattr(headers, 'get') else ''
        if 'zip' in content_type.lower():
            with NamedTemporaryFile(delete=False, suffix='.zip') as tmp_zip:
                shutil.copyfileobj(response, tmp_zip)
                tmp_zip_path = Path(tmp_zip.name)
            _write_zip_geotiff_from_file(tmp_zip_path, target)
            try: tmp_zip_path.unlink(missing_ok=True)
            except Exception: pass
        else:
            with target.open('wb') as out_f:
                shutil.copyfileobj(response, out_f)
    return ImageExportResult(path=target, task=task)

# -------- Streaming NDVI thresholds & classification (OOM-safe) --------

def _stream_histogram(ndvi_path: Path, bins: int = 2048, value_min: float = NDVI_PERCENTILE_MIN, value_max: float = NDVI_PERCENTILE_MAX) -> tuple[np.ndarray, np.ndarray, int]:
    hist = np.zeros(bins, dtype=np.int64)
    total = 0
    with rasterio.open(ndvi_path) as src:
        band = 1
        try:
            windows = list(src.block_windows(band))
        except Exception:
            step_x, step_y = 512, 512
            cols, rows = src.width, src.height
            windows = [((0, 0), Window(x, y, min(step_x, cols - x), min(step_y, rows - y)))
                       for y in range(0, rows, step_y)
                       for x in range(0, cols, step_x)]
        for _, win in windows:
            data = src.read(band, window=win, masked=True)
            if data.size == 0: continue
            arr = data.compressed()
            if arr.size == 0: continue
            arr = np.clip(arr, value_min, value_max)
            h, edges = np.histogram(arr, bins=bins, range=(value_min, value_max))
            hist += h.astype(np.int64, copy=False)
            total += int(arr.size)
    return hist, edges, total

def compute_percentile_thresholds_stream(ndvi_path: Path, n_classes: int) -> np.ndarray:
    bins = 2048
    hist, edges, total = _stream_histogram(ndvi_path, bins=bins)
    if total == 0 or hist.sum() == 0:
        raise ValueError(NDVI_MASK_EMPTY_ERROR)
    if np.count_nonzero(hist) <= 1:
        return np.array([], dtype=float)
    cdf = np.cumsum(hist) / float(hist.sum())
    cuts_needed = max(n_classes - 1, 0)
    if cuts_needed == 0:
        return np.array([], dtype=float)
    targets = [(i + 1) / n_classes for i in range(cuts_needed)]
    thresholds: List[float] = []
    for t in targets:
        idx = int(np.searchsorted(cdf, t, side='left'))
        idx = min(max(idx, 1), len(edges) - 1)
        val = float(edges[idx])
        if thresholds and not (val > thresholds[-1]):
            val = np.nextafter(thresholds[-1], math.inf)
        thresholds.append(val)
    return np.array(thresholds, dtype=float)

def classify_stream_to_file(ndvi_path: Path, thresholds: np.ndarray, out_path: Path) -> List[int]:
    with rasterio.open(ndvi_path) as src:
        profile = src.profile.copy()
        profile.update(dtype=rasterio.uint8, count=1, compress='lzw', nodata=0)
        with rasterio.open(out_path, 'w', **profile) as dst:
            unique_vals = set()
            band = 1
            try:
                windows = list(src.block_windows(band))
            except Exception:
                step_x, step_y = 512, 512
                cols, rows = src.width, src.height
                windows = [((0, 0), Window(x, y, min(step_x, cols - x), min(step_y, rows - y)))
                           for y in range(0, rows, step_y)
                           for x in range(0, cols, step_x)]
            for _, win in windows:
                ndvi = src.read(band, window=win, masked=True)
                if ndvi.size == 0:
                    continue
                data = ndvi.filled(np.nan)
                mask = ~ndvi.mask & np.isfinite(data)
                out = np.zeros(ndvi.shape, dtype=np.uint8)
                if thresholds.size:
                    cmp = data[..., None] > thresholds[None, None, :]
                    classes = cmp.sum(axis=-1, dtype=np.int16) + 1
                    out[mask] = classes[mask].astype(np.uint8)
                else:
                    out[mask] = 1
                dst.write(out, 1, window=win)
                unique_vals.update(np.unique(out[out > 0]).tolist())
    return sorted(unique_vals)

# -------- Vectorization & zonal stats --------

def _vectorize_to_shapefile(classified_path: Path, transform, crs, out_dir: Path) -> tuple[Path, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    shp_base = out_dir / 'zones'
    with rasterio.open(classified_path) as src:
        with shapefile.Writer(str(shp_base)) as writer:
            writer.autoBalance = 1
            writer.field('zone', 'N', decimal=0)
            writer.field('area_ha', 'F', decimal=4)
            for geom, value in shapes(src.read(1), mask=src.read_masks(1) > 0, transform=src.transform):
                zone_id = int(value)
                if zone_id <= 0: continue
                geom_shape = shape(geom)
                area_m2 = geom_shape.area
                area_ha = area_m2 / 10_000.0
                writer.record(zone_id, float(area_ha))
                writer.shape(mapping(geom_shape))
    shp_path = shp_base.with_suffix('.shp')
    cpg_path = shp_base.with_suffix('.cpg'); cpg_path.write_text('UTF-8')
    if crs:
        prj_path = shp_base.with_suffix('.prj')
        try: prj_path.write_text(CRS.from_user_input(crs).to_wkt())
        except Exception: prj_path.write_text('')
    vector_components: dict[str, str] = {}
    for ext in ['shp', 'dbf', 'shx', 'prj', 'cpg']:
        component = shp_base.with_suffix(f'.{ext}')
        if component.exists():
            vector_components[ext] = str(component)
    return shp_path, vector_components

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
            windows = [((0, 0), Window(x, y, min(step_x, cols - x), min(step_y, rows - y)))
                       for y in range(0, rows, step_y)
                       for x in range(0, cols, step_x)]
        px_area = abs(csrc.transform.a) * abs(csrc.transform.e)
        for _, win in windows:
            zones = csrc.read(band, window=win)
            ndvi = nsrc.read(band, window=win, masked=True).filled(np.nan)
            mask = (zones > 0) & np.isfinite(ndvi)
            if not np.any(mask):
                continue
            z = zones[mask].astype(np.int32, copy=False)
            v = ndvi[mask].astype(np.float32, copy=False)
            for zone_id in np.unique(z):
                m = z == zone_id
                vv = v[m]
                entry = stats.setdefault(int(zone_id), {'zone': int(zone_id), 'area_ha': 0.0, 'mean_ndvi': 0.0, 'min_ndvi': +1.0, 'max_ndvi': -1.0, 'pixel_count': 0})
                entry['pixel_count'] += int(vv.size)
                entry['area_ha'] += float(vv.size * px_area / 10_000.0)
                entry['mean_ndvi'] += float(vv.sum())
                entry['min_ndvi'] = float(min(entry['min_ndvi'], float(np.nanmin(vv))))
                entry['max_ndvi'] = float(max(entry['max_ndvi'], float(np.nanmax(vv))))
    for entry in stats.values():
        if entry['pixel_count'] > 0:
            entry['mean_ndvi'] = float(entry['mean_ndvi'] / entry['pixel_count'])
    return sorted(stats.values(), key=lambda d: d['zone'])

# -------- Local classification --------

def _classify_local_zones(ndvi_path: Path, *, working_dir: Path, n_classes: int, include_stats: bool) -> tuple[ZoneArtifacts, Dict[str, object]]:
    with rasterio.open(ndvi_path) as src:
        transform = src.transform
        crs = src.crs
        total_px = src.width * src.height

    thresholds = compute_percentile_thresholds_stream(ndvi_path, n_classes)
    raster_path = working_dir / 'zones_classified.tif'
    unique_zones = classify_stream_to_file(ndvi_path, thresholds, raster_path)

    vector_dir = working_dir / 'zones'
    shp_path, vector_components = _vectorize_to_shapefile(raster_path, transform, crs, vector_dir)

    stats_path = None
    zonal_stats = []
    if include_stats:
        stats_path = working_dir / 'zones_zonal_stats.csv'
        zonal_stats = _stream_zonal_stats(raster_path, ndvi_path)
        with open(stats_path, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=['zone', 'area_ha', 'mean_ndvi', 'min_ndvi', 'max_ndvi', 'pixel_count'])
            writer.writeheader()
            writer.writerows(zonal_stats)

    palette = list(ZONE_PALETTE[: max(1, min(len(unique_zones), len(ZONE_PALETTE)))])

    metadata: Dict[str, object] = {
        'percentile_thresholds': thresholds.tolist() if thresholds.size else [],
        'palette': palette,
        'zones': zonal_stats,
        'requested_zone_count': int(n_classes),
        'effective_zone_count': int(len(unique_zones)) if unique_zones else (1 if thresholds.size == 0 else n_classes),
        'final_zone_count': int(len(unique_zones)) if unique_zones else (1 if thresholds.size == 0 else n_classes),
        'classification_method': 'percentiles' if thresholds.size else 'single_class',
    }

    artifacts = ZoneArtifacts(
        raster_path=str(raster_path),
        mean_ndvi_path=str(ndvi_path),
        vector_path=str(shp_path),
        vector_components=vector_components,
        zonal_stats_path=str(stats_path) if stats_path else None,
        working_dir=str(working_dir),
    )
    return artifacts, metadata

# -------- Orchestration --------

def _ordered_months(months: Sequence[str]) -> List[str]:
    uniq = {}
    for raw in months:
        m = str(raw).strip()
        if not m: continue
        try: dt = datetime.strptime(m, '%Y-%m')
        except ValueError as exc: raise ValueError(f'Invalid month format: {raw}') from exc
        if m not in uniq: uniq[m] = dt
    return [k for k,_ in sorted(uniq.items(), key=lambda kv: kv[1])]

def _month_range_dates(months: Sequence[str]) -> tuple[date, date]:
    ordered = _ordered_months(months)
    if not ordered: raise ValueError('At least one month must be supplied')
    s = datetime.strptime(ordered[0], '%Y-%m')
    e = datetime.strptime(ordered[-1], '%Y-%m')
    start_day = date(s.year, s.month, 1)
    end_last_day = calendar.monthrange(e.year, e.month)[1]
    end_day = date(e.year, e.month, end_last_day)
    return start_day, end_day

def _months_from_dates(start_date: date, end_date: date) -> List[str]:
    if end_date < start_date: raise ValueError('end_date must be on or after start_date')
    months: List[str] = []
    cursor = date(start_date.year, start_date.month, 1)
    end_cursor = date(end_date.year, end_date.month, 1)
    while cursor <= end_cursor:
        months.append(cursor.strftime('%Y-%m'))
        cursor = (cursor.replace(year=cursor.year + 1, month=1) if cursor.month == 12 else cursor.replace(month=cursor.month + 1))
    return months

def _export_prefix(aoi_name: str, months: Sequence[str]) -> str:
    ordered = _ordered_months(months)
    start, end = ordered[0], ordered[-1]
    safe_name = sanitize_name(aoi_name or 'aoi')
    return f'zones/PROD_{start.replace('-', '')}_{end.replace('-', '')}_{safe_name}_zones'

def export_prefix(aoi_name: str, months: Sequence[str]) -> str:
    return _export_prefix(aoi_name, months)

def _resolve_geometry(aoi: Union[dict, ee.Geometry]) -> ee.Geometry:
    try:
        if isinstance(aoi, ee.Geometry): return aoi
    except TypeError:
        pass
    return gee.geometry_from_geojson(aoi)

def _stability_mask(cv_image: ee.Image, geometry: ee.Geometry, thresholds: Sequence[float], min_survival_ratio: float, scale: int) -> ee.Image:
    total = ee.Number(cv_image.reduceRegion(
        reducer=ee.Reducer.count(), geometry=geometry, scale=scale, bestEffort=True, tileScale=4, maxPixels=gee.MAX_PIXELS,
    ).values().get(0))
    threshold_list = ee.List([float(t) for t in thresholds])
    min_ratio = ee.Number(min_survival_ratio)
    def _mask_for_threshold(value):
        t = ee.Number(value)
        raw_mask = cv_image.lte(t)
        masked = ee.Image(raw_mask).selfMask()
        surviving = ee.Number(masked.reduceRegion(
            reducer=ee.Reducer.count(), geometry=geometry, scale=scale, bestEffort=True, tileScale=4, maxPixels=gee.MAX_PIXELS,
        ).values().get(0))
        ratio = surviving.divide(total.max(1))
        return ee.Image(ee.Algorithms.If(ratio.gte(min_ratio), masked, ee.Image(0).selfMask()))
    masks = threshold_list.map(_mask_for_threshold)
    combined = ee.ImageCollection.fromImages(masks).max()
    combined_masked = ee.Image(combined).selfMask()
    combined_count = ee.Number(combined_masked.reduceRegion(
        reducer=ee.Reducer.count(), geometry=geometry, scale=scale, bestEffort=True, tileScale=4, maxPixels=gee.MAX_PIXELS,
    ).values().get(0))
    pass_through = ee.Image(1)
    return ee.Image(ee.Algorithms.If(combined_count.lte(0), pass_through, combined_masked)).selfMask()

def _ndvi_temporal_stats(images: Sequence[ee.Image]) -> Mapping[str, ee.Image]:
    stats = temporal_stats(images, band_name='NDVI', rename_prefix='NDVI')
    return {'mean': stats['mean'], 'median': stats['median'], 'std': stats['std'], 'cv': stats['cv']}

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
    ordered_months = _ordered_months(months)
    composites, skipped_months, composite_metadata = _build_composite_series(
        geometry, ordered_months, start_date, end_date, cloud_prob_max
    )
    if not composites:
        raise ValueError(NDVI_MASK_EMPTY_ERROR)

    ndvi_images = [compute_ndvi_loose(img) for _, img in composites]
    ndvi_collection = ee.ImageCollection(ndvi_images)
    ndvi_stats = dict(_ndvi_temporal_stats(ndvi_collection))

    ndvi_mean = mean_from_collection_sum_count(ndvi_collection)
    first_ref = ee.Image(composites[0][1])
    ndvi_mean_native = (
        reproject_native_10m(ndvi_mean, first_ref, ref_band="B8", scale=DEFAULT_SCALE)
        .clip(geometry)
        .rename("NDVI_mean")
    )

    valid_pixel_count = int(
        ee.Number(
            ndvi_mean_native.mask()
            .reduceRegion(
                reducer=ee.Reducer.count(),
                geometry=geometry,
                scale=DEFAULT_SCALE,
                bestEffort=True,
                tileScale=4,
                maxPixels=gee.MAX_PIXELS,
            )
            .get("NDVI_mean")
        ).getInfo()
        or 0
    )
    if valid_pixel_count == 0:
        raise ValueError(NDVI_MASK_EMPTY_ERROR)

    ndvi_stats["mean"] = ndvi_mean_native


    stability_flag = APPLY_STABILITY if apply_stability_mask is None else bool(apply_stability_mask)
    if stability_flag:
        stability_image = _stability_mask(
            ndvi_stats['cv'], geometry, [0.5, 1.0, 1.5, 2.0], 0.0, DEFAULT_SCALE
        )
    else:
        stability_image = ee.Image(1)
    ndvi_stats['stability'] = stability_image

    workdir = _ensure_working_directory(working_dir)
    ndvi_path = workdir / 'mean_ndvi.tif'
    mean_export = _download_image_to_path(
        ndvi_stats['mean'].updateMask(stability_image), geometry, ndvi_path,
        params=_DownloadParams(crs=DEFAULT_EXPORT_CRS, scale=DEFAULT_SCALE),
    )
    ndvi_path = mean_export.path

    artifacts, local_metadata = _classify_local_zones(
        ndvi_path,
        working_dir=workdir,
        n_classes=n_classes,
        include_stats=include_stats,
    )

    metadata: Dict[str, object] = {
        'used_months': ordered_months,
        'skipped_months': skipped_months,
        'zone_method': 'ndvi_percentiles',
        'stability_mask_applied': stability_flag,
    }
    metadata.update(composite_metadata)
    metadata.update(local_metadata)
    metadata['downloaded_mean_ndvi'] = str(ndvi_path)
    metadata['mean_ndvi_export_task'] = _task_payload(mean_export.task)
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
    if n_classes < 2 or n_classes > 7:
        raise ValueError('n_classes must be between 2 and 7')
    if not months:
        raise ValueError('At least one month must be supplied')
    method_key = (method or '').strip().lower() or DEFAULT_METHOD
    if method_key != 'ndvi_percentiles':
        logger.warning('Unsupported method %s requested; falling back to ndvi_percentiles', method_key)
        method_key = 'ndvi_percentiles'
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
        smooth_radius_m=0,
        open_radius_m=0,
        close_radius_m=0,
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
    export_target: str = 'local',
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
        raise ValueError('end_date must be on or after start_date')

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
            raise ValueError('Either months or start/end dates must be supplied')
        start_dt = datetime.fromisoformat(start_date).date()
        end_dt = datetime.fromisoformat(end_date).date()
        months = _months_from_dates(start_dt, end_dt)

    ordered_months = _ordered_months(months)
    if start_date is None or end_date is None:
        s, e = _month_range_dates(ordered_months)
    else:
        s = datetime.fromisoformat(start_date).date()
        e = datetime.fromisoformat(end_date).date()

    include_stats_flag = bool(include_stats if include_stats is not None else include_zonal_stats)

    try:
        gee.initialize()
    except Exception:
        if not _allow_init_failure():
            raise
    geometry = geometry or _resolve_geometry(aoi_geojson)

    method_selection = (method or DEFAULT_METHOD).strip().lower()
    if method_selection != 'ndvi_percentiles':
        logger.warning('Unsupported method %s requested; falling back to ndvi_percentiles', method_selection)
        method_selection = 'ndvi_percentiles'

    artifacts, metadata = _prepare_selected_period_artifacts(
        aoi_geojson,
        geometry=geometry,
        working_dir=working_dir,
        months=ordered_months,
        start_date=s,
        end_date=e,
        cloud_prob_max=cloud_prob_max,
        n_classes=n_classes,
        cv_mask_threshold=cv_mask_threshold,
        apply_stability_mask=apply_stability_mask,
        min_mapping_unit_ha=mmu_ha,
        smooth_radius_m=0,
        open_radius_m=0,
        close_radius_m=0,
        simplify_tol_m=simplify_tol_m,
        simplify_buffer_m=simplify_buffer_m,
        method=method_selection,
        sample_size=DEFAULT_SAMPLE_SIZE,
        include_stats=include_stats_flag,
    )

    metadata = dict(metadata)
    metadata['zone_method'] = method_selection
    metadata = sanitize_for_json(metadata)
    used_months: list[str] = list(metadata.get('used_months', []))
    skipped: list[str] = list(metadata.get('skipped_months', []))
    if not used_months:
        raise ValueError('No valid Sentinel-2 scenes available for the selected period')

    prefix_base = _export_prefix(aoi_name, used_months)
    result: Dict[str, object] = {
        'paths': {
            'raster': artifacts.raster_path,
            'mean_ndvi': artifacts.mean_ndvi_path,
            'vectors': artifacts.vector_path,
            'vector_components': artifacts.vector_components,
            'zonal_stats': artifacts.zonal_stats_path if include_stats_flag else None,
        },
        'tasks': {},
        'prefix': prefix_base,
        'metadata': metadata,
        'artifacts': artifacts,
        'working_dir': artifacts.working_dir or str(working_dir),
    }

    palette = metadata.get('palette') if isinstance(metadata, dict) else None
    thresholds = metadata.get('percentile_thresholds') if isinstance(metadata, dict) else None
    if palette is not None:
        result['palette'] = palette
    if thresholds is not None:
        result['thresholds'] = thresholds

    export_target = (export_target or 'zip').strip().lower()
    if export_target not in {'zip', 'local'}:
        raise ValueError('Only local zone exports are supported in this workflow')

    return result

def _task_payload(task: ee.batch.Task | None) -> Dict[str, object]:
    if task is None:
        return {}
    payload: Dict[str, object] = {'id': getattr(task, 'id', None)}
    try:
        status = task.status() or {}
    except Exception:
        status = {}
    if status.get('state'):
        payload['state'] = status.get('state')
    destination = status.get('destination_uris') or []
    if destination:
        payload['destination_uris'] = destination
    error = status.get('error_message') or status.get('error_details')
    if error:
        payload['error'] = error
    return payload
