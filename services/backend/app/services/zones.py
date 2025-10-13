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
from app.utils.sanitization import sanitize_for_json

logger = logging.getLogger(__name__)

DEFAULT_CLOUD_PROB_MAX = 60
DEFAULT_N_CLASSES = 5
DEFAULT_MIN_MAPPING_UNIT_HA = 1
DEFAULT_SMOOTH_RADIUS_M = 0
DEFAULT_SIMPLIFY_TOL_M = 5
DEFAULT_SIMPLIFY_BUFFER_M = 3
DEFAULT_SMOOTH_RADIUS_PX = 1
DEFAULT_METHOD = "ndvi_linear"
DEFAULT_SAMPLE_SIZE = 4000
DEFAULT_SCALE = int(os.getenv("ZONES_SCALE_M", "10"))
DEFAULT_EXPORT_CRS = "EPSG:3857"
DEFAULT_CRS = DEFAULT_EXPORT_CRS

DEFAULT_NDVI_MIN = 0.35
DEFAULT_NDVI_MAX = 0.73

ZONE_PALETTE: Tuple[str, ...] = (
    "#112f1d", "#1b4d2a", "#2c6a39", "#3f8749", "#58a35d", "#80bf7d", "#b6dcb1",
)

NDVI_MASK_EMPTY_ERROR = (
    "No valid NDVI pixels across the selected months. Try a wider date range or relax cloud masking."
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
    except Exception as exc:
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
    if hasattr(image, "toFloat"):
        image = image.toFloat()

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
        "noData": -32768,
    }
    if params.crs:
        download_params["crs"] = params.crs
    url = image.getDownloadURL(download_params)

    with urlopen(url) as response:
        headers = getattr(response, "headers", None)
        content_type = headers.get("Content-Type", "") if headers and hasattr(headers, "get") else ""
        if "zip" in content_type.lower():
            with NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
                shutil.copyfileobj(response, tmp_zip)
                tmp_zip_path = Path(tmp_zip.name)
            with ZipFile(tmp_zip_path, "r") as archive:
                members = [m for m in archive.namelist() if m.lower().endswith((".tif", ".tiff"))]
                if not members:
                    raise ValueError("Zip archive did not contain a GeoTIFF file")
                first = members[0]
                with archive.open(first) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
            tmp_zip_path.unlink(missing_ok=True)
        else:
            with target.open("wb") as output:
                shutil.copyfileobj(response, output)
    return ImageExportResult(path=target, task=task)


def _download_vector_to_path(
    vectors: ee.FeatureCollection,
    target: Path,
    *, file_format: str,
) -> None:
    fmt = file_format.lower()
    if fmt not in {"geojson", "kml"}:
        raise ValueError("Unsupported vector format")

    params = {
        "filename": target.stem,
        "selectors": ["zone"],
        "crs": "EPSG:4326",
    }

    if fmt == "geojson" and target.suffix.lower() != ".geojson":
        target = target.with_suffix(".geojson")
    if fmt == "kml" and target.suffix.lower() != ".kml":
        target = target.with_suffix(".kml")
    target.parent.mkdir(parents=True, exist_ok=True)

    # Correct signature: getDownloadURL(filetype, params)
    url = vectors.getDownloadURL(fmt, params)

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
                tmp_zip_path.unlink(missing_ok=True)
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


def _export_prefix(aoi_name: str, months: Sequence[str]) -> str:
    ordered = _ordered_months(months)
    start, end = ordered[0], ordered[-1]
    safe_name = sanitize_name(aoi_name or "aoi")
    return f"zones/PROD_{start.replace('-', '')}_{end.replace('-', '')}_{safe_name}_zones"


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
    *, months: Sequence[str] | None = None,
    cloud_prob_max: int = DEFAULT_CLOUD_PROB_MAX,
) -> ee.Image:
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
        mask_1band = img.select("B8").mask().And(img.select("B4").mask())
        return ndvi.updateMask(mask_1band)

    ndvi_mean = monthly.map(_ndvi).mean().rename("NDVI_mean").toFloat().clip(geom)
    return ndvi_mean


def _make_thresholds(
    n_classes: int, *, ndvi_min: float | None, ndvi_max: float | None,
    custom_thresholds: List[float] | None,
) -> List[float]:
    if custom_thresholds:
        edges = [float(x) for x in custom_thresholds]
        if len(edges) != n_classes + 1:
            raise ValueError(f"custom_thresholds must have {n_classes+1} values")
        if any(edges[i+1] <= edges[i] for i in range(len(edges)-1)):
            raise ValueError("custom_thresholds must be strictly increasing")
        return edges

    lo = DEFAULT_NDVI_MIN if ndvi_min is None else float(ndvi_min)
    hi = DEFAULT_NDVI_MAX if ndvi_max is None else float(ndvi_max)
    if hi <= lo:
        hi = lo + 1e-6
    step = (hi - lo) / n_classes
    return [lo + i * step for i in range(n_classes + 1)]


def _classify_smooth_and_polygonize(
    ndvi_mean_native: ee.Image,
    geom: ee.Geometry,
    *, n_zones: int, mmu_ha: float, smooth_radius_px: int, thresholds: List[float],
):
    ndvi = ndvi_mean_native.rename("NDVI_mean").clip(geom)

    def classify(img: ee.Image, edges: List[float]) -> ee.Image:
        zones = ee.Image.constant(0)
        for i in range(len(edges) - 1):
            lower = edges[i]
            upper = edges[i + 1]
            zones = zones.where(img.gte(lower).And(img.lt(upper)), i + 1)
        zones = zones.where(img.gte(edges[-2]).And(img.lte(edges[-1])), len(edges) - 1)
        return zones.rename("zone").updateMask(img.mask()).toInt8()

    cls_raw = classify(ndvi, thresholds)

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
        reducer=ee.Reducer.first(),
        bestEffort=True,
        maxPixels=gee.MAX_PIXELS,
    )

    cls_mmu = cls_mmu.set({
        "thresholds": thresholds,
        "zones": n_zones,
        "mode": "linear_fixed",
    })

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
    *, geometry: ee.Geometry, working_dir: Path, months: Sequence[str],
    start_date: date | None, end_date: date | None, cloud_prob_max: int, n_classes: int,
    min_mapping_unit_ha: float, smooth_radius_m: float,
    simplify_tol_m: float, simplify_buffer_m: float,
    method: str, sample_size: int, include_stats: bool,
    ndvi_min: float | None, ndvi_max: float | None, custom_thresholds: List[float] | None,
) -> tuple[ZoneArtifacts, Dict[str, object]]:

    try:
        ordered_months = _ordered_months(months)

        # If start/end not provided, derive from months
        if start_date is None or end_date is None:
            start_dt = datetime.strptime(ordered_months[0], "%Y-%m")
            end_dt = datetime.strptime(ordered_months[-1], "%Y-%m")
            start_date = date(start_dt.year, start_dt.month, 1)
            import calendar as _cal
            end_date = date(end_dt.year, end_dt.month, _cal.monthrange(end_dt.year, end_dt.month)[1])

        # mean NDVI
        ndvi_mean_native = _build_mean_ndvi_for_zones(
            geometry, start_date, end_date, months=ordered_months, cloud_prob_max=cloud_prob_max,
        )

        # thresholds
        thresholds = _make_thresholds(
            n_classes, ndvi_min=ndvi_min, ndvi_max=ndvi_max, custom_thresholds=custom_thresholds
        )

        # classify & polygonize
        classified_image, vectors = _classify_smooth_and_polygonize(
            ndvi_mean_native,
            geometry,
            n_zones=n_classes,
            mmu_ha=min_mapping_unit_ha,
            smooth_radius_px=max(0, int(round(smooth_radius_m / 10))),
            thresholds=thresholds,
        )

        workdir = _ensure_working_directory(working_dir)

        ndvi_path = workdir / "NDVI_mean.tif"
        mean_export = _download_image_to_path(
            ndvi_mean_native, geometry, ndvi_path, params=_DownloadParams(crs=DEFAULT_EXPORT_CRS, scale=DEFAULT_SCALE),
        )
        ndvi_path = mean_export.path

        classified_path = workdir / "zones.tif"
        classified_export = _download_image_to_path(
            classified_image, geometry, classified_path, params=_DownloadParams(crs=DEFAULT_EXPORT_CRS, scale=DEFAULT_SCALE),
        )
        classified_path = classified_export.path

        vectors_reprojected = ee.FeatureCollection(
            vectors.map(lambda f: f.setGeometry(f.geometry().transform("EPSG:4326", 0.1)))
        )

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
                    fieldnames=["zone", "area_ha", "mean_ndvi", "min_ndvi", "max_ndvi", "pixel_count"],
                )
                writer.writeheader()
                writer.writerows(zonal_stats)

        palette = list(ZONE_PALETTE[: max(1, min(n_classes, len(ZONE_PALETTE)))])
        metadata: Dict[str, Any] = {
            "used_months": ordered_months,
            "skipped_months": [],
            "palette": palette,
            "requested_zone_count": int(n_classes),
            "thresholds": thresholds,
            "zones": zonal_stats,
            "classification_mode": "linear_fixed",
        }

        artifacts = ZoneArtifacts(
            raster_path=str(classified_path),
            mean_ndvi_path=str(ndvi_path),
            vector_path=str(geojson_path),
            vector_components={"geojson": str(geojson_path), "kml": str(kml_path)},
            zonal_stats_path=str(stats_path) if stats_path else None,
            working_dir=str(workdir),
        )

        metadata["downloaded_mean_ndvi"] = str(ndvi_path)
        metadata["mean_ndvi_export_task"] = _task_payload(mean_export.task)
        metadata["classified_export_task"] = _task_payload(classified_export.task)
        return artifacts, metadata

    except ee.ee_exception.EEException as ee_err:
        msg = str(ee_err)
        raise ValueError(f"Earth Engine error: {msg}")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Unexpected error while preparing NDVI zones: {exc}")


def build_zone_artifacts(
    aoi_geojson: Union[dict, ee.Geometry],
    *, months: Sequence[str],
    cloud_prob_max: int = DEFAULT_CLOUD_PROB_MAX,
    n_classes: int = DEFAULT_N_CLASSES,
    min_mapping_unit_ha: float = DEFAULT_MIN_MAPPING_UNIT_HA,
    smooth_radius_m: float = DEFAULT_SMOOTH_RADIUS_M,
    simplify_tolerance_m: float = DEFAULT_SIMPLIFY_TOL_M,
    simplify_buffer_m: float = DEFAULT_SIMPLIFY_BUFFER_M,
    method: str = DEFAULT_METHOD,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    include_stats: bool = True,
    ndvi_min: float | None = None,
    ndvi_max: float | None = None,
    custom_thresholds: List[float] | None = None,
) -> ZoneArtifacts:
    if n_classes not in (3, 5):
        raise ValueError("n_classes must be 3 or 5")
    if not months:
        raise ValueError("At least one month must be supplied")

    try:
        gee.initialize()
    except Exception:
        if not _allow_init_failure():
            raise

    geometry = _resolve_geometry(aoi_geojson)
    ordered = _ordered_months(months)
    start_dt = datetime.strptime(ordered[0], "%Y-%m").date().replace(day=1)
    end_dt_src = datetime.strptime(ordered[-1], "%Y-%m")
    end_dt = date(end_dt_src.year, end_dt_src.month, calendar.monthrange(end_dt_src.year, end_dt_src.month)[1])

    working_dir = _ensure_working_directory(None)

    artifacts, _metadata = _prepare_selected_period_artifacts(
        aoi_geojson,
        geometry=geometry,
        working_dir=working_dir,
        months=ordered,
        start_date=start_dt,
        end_date=end_dt,
        cloud_prob_max=cloud_prob_max,
        n_classes=n_classes,
        min_mapping_unit_ha=min_mapping_unit_ha,
        smooth_radius_m=smooth_radius_m,
        simplify_tol_m=simplify_tolerance_m,
        simplify_buffer_m=simplify_buffer_m,
        method=method,
        sample_size=DEFAULT_SAMPLE_SIZE,
        include_stats=include_stats,
        ndvi_min=ndvi_min,
        ndvi_max=ndvi_max,
        custom_thresholds=custom_thresholds,
    )
    return artifacts


def export_selected_period_zones(
    aoi_geojson: dict, aoi_name: str, months: list[str],
    *, geometry: ee.Geometry | None = None,
    start_date: str | None = None, end_date: str | None = None,
    cloud_prob_max: int = DEFAULT_CLOUD_PROB_MAX,
    n_classes: int = DEFAULT_N_CLASSES,
    mmu_ha: float = DEFAULT_MIN_MAPPING_UNIT_HA,
    smooth_radius_m: int = DEFAULT_SMOOTH_RADIUS_M,
    simplify_tol_m: int = DEFAULT_SIMPLIFY_TOL_M,
    simplify_buffer_m: int = DEFAULT_SIMPLIFY_BUFFER_M,
    export_target: str = "zip", destination: str | None = None,
    gcs_bucket: str | None = None, gcs_prefix: str | None = None,
    include_zonal_stats: bool = True, method: str | None = None,
    ndvi_min: float | None = None, ndvi_max: float | None = None,
    custom_thresholds: List[float] | None = None,
) -> Dict[str, Any]:
    working_dir = _ensure_working_directory(None)
    aoi = _to_ee_geometry(aoi_geojson)
    geometry = geometry or aoi

    if n_classes not in (3, 5):
        raise ValueError("n_classes must be 3 or 5")

    if destination is not None:
        export_target = destination

    def _coerce_date_any(d):
        if d is None:
            return None
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
        if end_dt < start_dt:
            raise ValueError("end_date must be on or after start_date")
        months = []
        cursor = date(start_dt.year, start_dt.month, 1)
        end_cursor = date(end_dt.year, end_dt.month, 1)
        while cursor <= end_cursor:
            months.append(cursor.strftime("%Y-%m"))
            if cursor.month == 12:
                cursor = date(cursor.year + 1, 1, 1)
            else:
                cursor = date(cursor.year, cursor.month + 1, 1)

    ordered_months = _ordered_months(months)

    try:
        gee.initialize()
    except Exception:
        if not _allow_init_failure():
            raise

    geometry = geometry or _resolve_geometry(aoi_geojson)

    artifacts, metadata = _prepare_selected_period_artifacts(
        aoi_geojson,
        geometry=geometry,
        working_dir=working_dir,
        months=ordered_months,
        start_date=None if start_date is None else _coerce_date_any(start_date),
        end_date=None if end_date is None else _coerce_date_any(end_date),
        cloud_prob_max=cloud_prob_max,
        n_classes=n_classes,
        min_mapping_unit_ha=mmu_ha,
        smooth_radius_m=smooth_radius_m,
        simplify_tol_m=simplify_tol_m,
        simplify_buffer_m=simplify_buffer_m,
        method=(method or DEFAULT_METHOD),
        sample_size=DEFAULT_SAMPLE_SIZE,
        include_stats=bool(include_zonal_stats),
        ndvi_min=ndvi_min,
        ndvi_max=ndvi_max,
        custom_thresholds=custom_thresholds,
    )

    metadata = sanitize_for_json(dict(metadata))
    used_months: list[str] = list(metadata.get("used_months", [])) or ordered_months
    prefix_base = _export_prefix(aoi_name, used_months)

    result: Dict[str, Any] = {
        "paths": {
            "raster": artifacts.raster_path,
            "mean_ndvi": artifacts.mean_ndvi_path,
            "vectors": artifacts.vector_path,
            "vector_components": artifacts.vector_components,
            "zonal_stats": artifacts.zonal_stats_path if include_zonal_stats else None,
        },
        "tasks": {},
        "prefix": prefix_base,
        "metadata": metadata,
        "artifacts": artifacts,
        "working_dir": artifacts.working_dir or str(working_dir),
        "palette": ZONE_PALETTE[:n_classes],
        "thresholds": metadata.get("thresholds"),
    }

    export_target = (export_target or "zip").strip().lower()
    if export_target not in {"zip", "local"}:
        raise ValueError("Only local/zip zone exports are supported in this workflow")

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
