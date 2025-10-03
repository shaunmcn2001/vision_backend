"""Sentinel-2 NDVI percentile zoning utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import ee
from ee.ee_exception import EEException

from app import gee
from app.services import zonesold as _legacy_zones

CLOUD_PROB_MAX = 40
N_CLASSES = 5
CV_THRESHOLD = 0.5
MIN_MAPPING_UNIT_HA = 1.5
SMOOTH_RADIUS_M = 30
OPEN_RADIUS_M = 10
CLOSE_RADIUS_M = 10
SIMPLIFY_TOL_M = 5
SIMPLIFY_BUFFER_M = 3
SAMPLE_SIZE = 8000
SCALE = 10
EXPORT_CRS = "EPSG:3857"

# Compatibility aliases preserved from the historical service module.  The API
# routers and older tests still import these names, so keep them pointing at the
# updated defaults to avoid AttributeError crashes when the module is imported.
DEFAULT_CLOUD_PROB_MAX = CLOUD_PROB_MAX
DEFAULT_N_CLASSES = N_CLASSES
DEFAULT_CV_THRESHOLD = CV_THRESHOLD
DEFAULT_MIN_MAPPING_UNIT_HA = MIN_MAPPING_UNIT_HA
DEFAULT_SMOOTH_RADIUS_M = SMOOTH_RADIUS_M
DEFAULT_OPEN_RADIUS_M = OPEN_RADIUS_M
DEFAULT_CLOSE_RADIUS_M = CLOSE_RADIUS_M
DEFAULT_SIMPLIFY_TOL_M = SIMPLIFY_TOL_M
DEFAULT_SIMPLIFY_BUFFER_M = SIMPLIFY_BUFFER_M
DEFAULT_SCALE = SCALE
DEFAULT_CRS = EXPORT_CRS
NDVI_PERCENTILE_MIN = 0.0
NDVI_PERCENTILE_MAX = 0.6
STABILITY_FALLBACKS = [0.5, 1.0, 1.5, 2.0]
MIN_STABILITY_SURVIVAL_RATIO = 0.0
ZONE_PALETTE = (
    "#d73027",
    "#fc8d59",
    "#fee08b",
    "#d9ef8b",
    "#91bfdb",
    "#4575b4",
)


@dataclass(frozen=True)
class ZoneArtifacts:
    """Container for zone outputs and diagnostic imagery."""

    zone_image: ee.Image
    zone_vectors: ee.FeatureCollection
    zonal_stats: ee.FeatureCollection | None
    geometry: ee.Geometry


def _force_ndvi_mean_band(image: ee.Image) -> ee.Image:
    """Normalise NDVI mean band naming to exactly ``NDVI_mean``."""

    b = ee.String(image.bandNames().get(0))
    fixed = ee.Algorithms.If(
        b.regex("(?i)^ndv[i1l]_?\\s*mean$"),
        "NDVI_mean",
        b,
    )
    return image.rename(ee.String(fixed))


def _ensure_number(x: Any, context: str = "") -> ee.Number:
    """Ensure a plain ``ee.Number`` is supplied, not an ``ee.Image``."""

    message = (
        "Expected Number for "
        f"{context}"
        "; got Image. Never use ee.Image.constant(...) with an Image."
    )
    ee_error_cls = getattr(ee, "Error", None)

    if ee_error_cls is not None:
        t = ee.String(ee.Algorithms.ObjectType(x))
        return ee.Number(
            ee.Algorithms.If(
                t.compareTo("Image").eq(0),
                ee_error_cls(message),
                x,
            )
        )

    ee_image_type = getattr(ee, "Image", None)
    if isinstance(ee_image_type, type) and isinstance(x, ee_image_type):
        raise EEException(message)

    if hasattr(x, "updateMask") and hasattr(x, "reduceRegion"):
        raise EEException(message)

    try:
        return ee.Number(x)
    except Exception as exc:  # pragma: no cover - defensive guard for fakes
        raise EEException(message) from exc


def _to_geometry(aoi_geojson_or_geom: Union[dict, ee.Geometry]) -> ee.Geometry:
    """Normalise GeoJSON/feature inputs to :class:`ee.Geometry`."""

    ee_geometry_type = getattr(ee, "Geometry", None)
    if isinstance(ee_geometry_type, type) and isinstance(
        aoi_geojson_or_geom, ee_geometry_type
    ):
        return aoi_geojson_or_geom

    if not isinstance(aoi_geojson_or_geom, Mapping):
        raise TypeError("AOI must be an ee.Geometry or a GeoJSON mapping.")

    geojson = aoi_geojson_or_geom
    if geojson.get("type") == "FeatureCollection":
        features = geojson.get("features", [])
        if not features:
            raise ValueError("GeoJSON FeatureCollection contains no features.")
        geometries = [
            ee.Feature(feat).geometry()
            for feat in features
            if isinstance(feat, Mapping)
        ]
        return ee.FeatureCollection(geometries).geometry()

    if geojson.get("type") == "Feature":
        return ee.Feature(geojson).geometry()

    return ee.Geometry(geojson)


def _month_start(month: str) -> date:
    year, month_num = month.split("-")
    return date(int(year), int(month_num), 1)


def _parse_months(months: Sequence[str]) -> Tuple[List[str], str, str, List[Mapping[str, Any]]]:
    if not months:
        raise ValueError("At least one month (YYYY-MM) must be provided.")

    unique_months = sorted({m: None for m in months}.keys())
    start = _month_start(unique_months[0])
    last = _month_start(unique_months[-1])
    end_exclusive = (last.replace(day=1) + timedelta(days=32)).replace(day=1)

    month_ranges: List[Mapping[str, Any]] = []
    for month in unique_months:
        start_date = _month_start(month)
        end_date = (start_date + timedelta(days=32)).replace(day=1)
        month_ranges.append(
            {
                "label": month,
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            }
        )

    return unique_months, start.isoformat(), end_exclusive.isoformat(), month_ranges


def _mask_s2_image(image: ee.Image, cloud_prob_max: int) -> ee.Image:
    qa = image.select("QA60")
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    qa_mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
        qa.bitwiseAnd(cirrus_bit_mask).eq(0)
    )

    scl = image.select("SCL")
    shadow_mask = scl.neq(3).And(scl.neq(11))

    cloud_prob = ee.Image(image.get("cloud_prob"))
    prob_mask = cloud_prob.select("probability").lte(_ensure_number(cloud_prob_max))

    mask = qa_mask.And(shadow_mask).And(prob_mask)

    scaled = image.select(["B2", "B3", "B4", "B8"]).divide(10000)
    return scaled.updateMask(mask).copyProperties(image, image.propertyNames())


def _build_masked_s2_collection(
    geometry: ee.Geometry,
    start_date: str,
    end_date: str,
    cloud_prob_max: int,
) -> ee.ImageCollection:
    s2_sr = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterDate(
        start_date, end_date
    ).filterBounds(geometry)

    s2_clouds = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY").filterDate(
        start_date, end_date
    ).filterBounds(geometry)

    joined = ee.ImageCollection(
        ee.Join.saveFirst(matchKey="cloud_prob").apply(
            primary=s2_sr,
            secondary=s2_clouds,
            condition=ee.Filter.equals(
                leftField="system:index", rightField="system:index"
            ),
        )
    )

    def _apply_mask(img: ee.Image) -> ee.Image:
        masked = _mask_s2_image(img, cloud_prob_max)
        return masked.clip(geometry)

    return ee.ImageCollection(joined.map(_apply_mask))


def _build_composite_collection(
    base_collection: ee.ImageCollection,
    month_ranges: Sequence[Mapping[str, Any]],
    use_monthly: bool,
) -> Tuple[ee.ImageCollection, List[str]]:
    if not use_monthly:
        return base_collection, []

    zero_mask = ee.Image.constant(0)
    empty_template = ee.Image.constant([0, 0, 0, 0]).rename(["B2", "B3", "B4", "B8"]).updateMask(
        zero_mask
    )

    def _monthly_image(month_dict: Mapping[str, Any]) -> ee.Image:
        md = ee.Dictionary(month_dict)
        start = ee.Date(md.get("start"))
        end = ee.Date(md.get("end"))
        label = ee.String(md.get("label"))
        coll = base_collection.filterDate(start, end)
        count = ee.Number(coll.size())
        composite = ee.Image(
            ee.Algorithms.If(
                count.eq(0),
                empty_template,
                coll.median(),
            )
        )
        return (
            composite.set("month", label)
            .set("image_count", count)
            .set("system:time_start", start.millis())
        )

    composites = ee.ImageCollection.fromImages(
        ee.List(list(month_ranges)).map(_monthly_image)
    )
    filtered = composites.filter(ee.Filter.gt("image_count", 0))

    skipped = ee.List(
        composites
        .filter(ee.Filter.eq("image_count", 0))
        .aggregate_array("month")
    ).map(lambda m: ee.String(m)).getInfo()

    return filtered, [str(m) for m in skipped]


def _ndvi_collection(collection: ee.ImageCollection) -> ee.ImageCollection:
    def _with_ndvi(image: ee.Image) -> ee.Image:
        ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
        projection = image.select("B8").projection()
        ndvi = ndvi.setDefaultProjection(projection)
        return ndvi.copyProperties(image, image.propertyNames())

    return collection.map(_with_ndvi)


def _ndvi_statistics(
    ndvi_images: ee.ImageCollection,
    geometry: ee.Geometry,
) -> Tuple[ee.Image, ee.Image, ee.Image, ee.Image]:
    ndvi_mean = _force_ndvi_mean_band(ndvi_images.reduce(ee.Reducer.mean()))
    ndvi_median = ndvi_images.reduce(ee.Reducer.median()).rename("NDVI_median")
    ndvi_std = ndvi_images.reduce(ee.Reducer.stdDev()).rename("NDVI_stdDev")

    divisor = ndvi_mean.where(
        ndvi_mean.lte(0),
        _ensure_number(1e-6, "cv_divisor"),
    )
    ndvi_cv = ndvi_std.divide(divisor).rename("NDVI_cv")
    ndvi_cv = ndvi_cv.where(ndvi_mean.lte(0), 0)

    pixel_count = ee.Number(
        ndvi_mean.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=geometry,
            scale=SCALE,
            bestEffort=True,
            maxPixels=1_000_000_000,
        ).get("NDVI_mean")
    ).getInfo()

    if not pixel_count:
        raise ValueError(
            "No NDVI pixels were available for the selected period. "
            "Try expanding the date range or relaxing the cloud filtering threshold."
        )

    return ndvi_mean, ndvi_median, ndvi_std, ndvi_cv


def _stability_mask(
    ndvi_cv: ee.Image,
    geometry: ee.Geometry,
    threshold: float,
    apply: bool,
) -> Tuple[ee.Image, Dict[str, Any]]:
    projection = ndvi_cv.projection()
    ones = ee.Image.constant(1).setDefaultProjection(projection)

    if not apply:
        return ones, {
            "applied": False,
            "selected_threshold": None,
            "survival_ratio": 1.0,
            "tried_thresholds": [],
        }

    total_count = ee.Number(
        ndvi_cv.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=geometry,
            scale=SCALE,
            bestEffort=True,
            maxPixels=1_000_000_000,
        ).get("NDVI_cv")
    ).getInfo()

    if not total_count:
        return ones, {
            "applied": False,
            "selected_threshold": None,
            "survival_ratio": 1.0,
            "tried_thresholds": [],
        }

    candidate_thresholds: List[float] = []
    if threshold not in candidate_thresholds:
        candidate_thresholds.append(float(threshold))
    for fallback in STABILITY_FALLBACKS:
        if fallback not in candidate_thresholds:
            candidate_thresholds.append(float(fallback))

    chosen = None
    ratio = 0.0
    tried: List[Dict[str, float]] = []

    for candidate in candidate_thresholds:
        mask = ndvi_cv.lte(_ensure_number(candidate, "stability_threshold"))
        survivors = ee.Number(
            ndvi_cv.updateMask(mask).reduceRegion(
                reducer=ee.Reducer.count(),
                geometry=geometry,
                scale=SCALE,
                bestEffort=True,
                maxPixels=1_000_000_000,
            ).get("NDVI_cv")
        ).getInfo() or 0
        current_ratio = survivors / float(total_count)
        tried.append({"threshold": candidate, "survival_ratio": current_ratio})
        if survivors > 0 and current_ratio >= MIN_STABILITY_SURVIVAL_RATIO:
            chosen = candidate
            ratio = current_ratio
            break

    if chosen is None:
        return ones, {
            "applied": False,
            "selected_threshold": None,
            "survival_ratio": 1.0,
            "tried_thresholds": tried,
        }

    mask_image = ones.updateMask(
        ndvi_cv.lte(_ensure_number(chosen, "stability_threshold"))
    )

    return mask_image, {
        "applied": True,
        "selected_threshold": chosen,
        "survival_ratio": ratio,
        "tried_thresholds": tried,
    }


def _ndvi_percentile_thresholds(
    ndvi_mean: ee.Image,
    geometry: ee.Geometry,
    n_classes: int,
) -> Tuple[ee.List, List[float]]:
    if n_classes < 1:
        raise ValueError("Number of classes must be at least 1.")

    if n_classes == 1:
        return ee.List([]), []

    percent_values = [(i * 100.0) / n_classes for i in range(1, n_classes)]
    reducer = ee.Reducer.percentile(percent_values)
    dictionary = ee.Dictionary(
        ndvi_mean.reduceRegion(
            reducer=reducer,
            geometry=geometry,
            scale=SCALE,
            bestEffort=True,
            maxPixels=1_000_000_000,
            tileScale=4,
        )
    )

    size = ee.Number(dictionary.size()).getInfo()
    if not size:
        raise ValueError(
            "All pixels were masked out by the stability threshold when computing NDVI percentiles. "
            "Try lowering the coefficient of variation threshold, expanding the selected months."
        )

    values = ee.List(dictionary.values(dictionary.keys())).sort()
    thresholds_info = values.getInfo()

    if thresholds_info is None:
        raise ValueError(
            "All pixels were masked out by the stability threshold when computing NDVI percentiles. "
            "Try lowering the coefficient of variation threshold, expanding the selected months."
        )

    thresholds = [float(v) for v in thresholds_info]
    return values, thresholds


def _classify_percentiles(
    ndvi_mean: ee.Image,
    thresholds: ee.List,
) -> ee.Image:
    projection = ndvi_mean.projection()
    base = ee.Image.constant(0).setDefaultProjection(projection)

    def _apply(threshold, image):
        threshold_number = _ensure_number(threshold, "ndvi_percentile")
        increment = ndvi_mean.gt(threshold_number)
        return ee.Image(image).add(increment)

    classified = ee.Image(thresholds.iterate(_apply, base))
    classified = classified.add(1)
    classified = classified.updateMask(ndvi_mean.mask())
    return classified.rename("zone").toInt16()


def _clean_zones(
    zones: ee.Image,
    geometry: ee.Geometry,
    smooth_radius_m: float,
    open_radius_m: float,
    close_radius_m: float,
    min_mapping_unit_ha: float,
) -> ee.Image:
    cleaned = ee.Image(zones)

    if smooth_radius_m > 0:
        cleaned = ee.Image(
            cleaned.focal_mode(
                radius=smooth_radius_m,
                kernelType="circle",
                units="meters",
            )
        )

    if open_radius_m > 0:
        cleaned = ee.Image(
            cleaned.focal_min(
                radius=open_radius_m,
                kernelType="circle",
                units="meters",
            ).focal_max(
                radius=open_radius_m,
                kernelType="circle",
                units="meters",
            )
        )

    if close_radius_m > 0:
        cleaned = ee.Image(
            cleaned.focal_max(
                radius=close_radius_m,
                kernelType="circle",
                units="meters",
            ).focal_min(
                radius=close_radius_m,
                kernelType="circle",
                units="meters",
            )
        )

    cleaned = ee.Image(cleaned.updateMask(cleaned.gt(0)).rename("zone"))

    pixel_area = ee.Image.pixelArea()
    min_area_m2 = float(min_mapping_unit_ha) * 10_000.0

    connected_area = pixel_area.addBands(cleaned).reduceConnectedComponents(
        reducer=ee.Reducer.sum(),
        labelBand="zone",
        maxSize=1 << 22,
    ).rename("component_area")

    small_components = connected_area.lt(min_area_m2)

    fill_radius = max(smooth_radius_m, open_radius_m, close_radius_m, 1)
    majority = ee.Image(
        cleaned.focal_mode(
            radius=fill_radius,
            kernelType="circle",
            units="meters",
        )
    )

    filled = ee.Image(cleaned.where(small_components, majority))
    filled = ee.Image(filled.updateMask(filled.gt(0)).rename("zone"))
    return ee.Image(filled).clip(geometry).toInt16()


def _vectorize_zones(
    zone_image: ee.Image,
    geometry: ee.Geometry,
    simplify_tolerance_m: float,
    simplify_buffer_m: float,
) -> ee.FeatureCollection:
    mask = zone_image.updateMask(zone_image.neq(0))

    vectors = mask.reduceToVectors(
        geometry=geometry,
        scale=SCALE,
        geometryType="polygon",
        eightConnected=True,
        labelProperty="zone",
        reducer=ee.Reducer.first(),
        maxPixels=1_000_000_000,
    )

    tolerance = float(simplify_tolerance_m)
    buffer_distance = float(simplify_buffer_m)

    def _decorate(feature: ee.Feature) -> ee.Feature:
        geom = feature.geometry().intersection(geometry, 1)
        simplified = geom.simplify(tolerance)
        buffered = simplified.buffer(buffer_distance)
        clipped = buffered.intersection(geometry, 1)
        area_m2 = clipped.area(maxError=1)
        area_ha = area_m2.divide(10_000.0)

        system_index = ee.String(feature.get("system:index"))
        numeric_index = system_index.replace("[^0-9]", "")
        fallback = ee.String(ee.Number(feature.get("zone")).format("%d"))
        index_string = ee.String(
            ee.Algorithms.If(numeric_index.length().gt(0), numeric_index, fallback)
        )
        zone_id = ee.Number.parse(index_string).add(1)

        return (
            feature.setGeometry(clipped)
            .set("zone", ee.Number(feature.get("zone")).int())
            .set("zone_id", zone_id.int())
            .set("area_m2", area_m2)
            .set("area_ha", area_ha)
        )

    return vectors.map(_decorate)


def _zonal_statistics(
    vectors: ee.FeatureCollection,
    stats_image: ee.Image,
) -> ee.FeatureCollection:
    reducer = ee.Reducer.mean()
    return stats_image.reduceRegions(
        collection=vectors,
        reducer=reducer,
        scale=SCALE,
        bestEffort=True,
        maxPixels=1_000_000_000,
    )


def build_zone_artifacts(
    aoi_geojson_or_geom: Union[dict, ee.Geometry],
    *,
    months: Sequence[str],
    cloud_prob_max: int = CLOUD_PROB_MAX,
    n_classes: int = N_CLASSES,
    cv_mask_threshold: float = CV_THRESHOLD,
    apply_stability_mask: bool | None = True,
    min_mapping_unit_ha: float = MIN_MAPPING_UNIT_HA,
    smooth_radius_m: float = SMOOTH_RADIUS_M,
    open_radius_m: float = OPEN_RADIUS_M,
    close_radius_m: float = CLOSE_RADIUS_M,
    simplify_tolerance_m: float = SIMPLIFY_TOL_M,
    simplify_buffer_m: float = SIMPLIFY_BUFFER_M,
    include_stats: bool = True,
) -> ZoneArtifacts:
    """Compute NDVI percentile zones and derived artefacts.

    Example
    -------
    >>> artifacts = build_zone_artifacts(aoi, months=["2025-07", "2025-08"])
    >>> artifacts.zone_image
    """

    geometry = _to_geometry(aoi_geojson_or_geom)
    month_list, start_date, end_date, month_ranges = _parse_months(months)

    base_collection = _build_masked_s2_collection(
        geometry=geometry,
        start_date=start_date,
        end_date=end_date,
        cloud_prob_max=cloud_prob_max,
    )

    use_monthly = len(month_list) >= 3
    composites, skipped_months = _build_composite_collection(
        base_collection=base_collection,
        month_ranges=month_ranges,
        use_monthly=use_monthly,
    )

    if not ee.Number(composites.size()).getInfo():
        raise ValueError(
            "No NDVI pixels were available for the selected period. "
            "Try expanding the date range or relaxing the cloud filtering threshold."
        )

    ndvi_images = _ndvi_collection(composites)
    ndvi_mean, ndvi_median, ndvi_std, ndvi_cv = _ndvi_statistics(
        ndvi_images, geometry
    )

    stability_apply = True if apply_stability_mask is None else bool(apply_stability_mask)

    stability_mask, stability_info = _stability_mask(
        ndvi_cv,
        geometry,
        threshold=float(cv_mask_threshold),
        apply=stability_apply,
    )

    ndvi_mean_clamped = ndvi_mean.clamp(
        _ensure_number(NDVI_PERCENTILE_MIN, "ndvi_percentile_min"),
        _ensure_number(NDVI_PERCENTILE_MAX, "ndvi_percentile_max"),
    )

    percentile_thresholds, thresholds_info = _ndvi_percentile_thresholds(
        ndvi_mean_clamped,
        geometry,
        n_classes,
    )

    zone_raw = _classify_percentiles(ndvi_mean_clamped, percentile_thresholds)
    zone_masked = zone_raw.updateMask(stability_mask)

    zone_clean = _clean_zones(
        zone_masked,
        geometry,
        smooth_radius_m,
        open_radius_m,
        close_radius_m,
        min_mapping_unit_ha,
    )

    skipped_set = set(skipped_months)
    used_months = [m for m in month_list if m not in skipped_set]

    stats_image = ee.Image.cat([ndvi_mean, ndvi_median, ndvi_std, ndvi_cv])

    zone_vectors = _vectorize_zones(
        zone_clean,
        geometry,
        simplify_tolerance_m,
        simplify_buffer_m,
    )

    zonal_stats = (
        _zonal_statistics(zone_vectors, stats_image) if include_stats else None
    )

    palette = [ZONE_PALETTE[i % len(ZONE_PALETTE)] for i in range(max(1, n_classes))]

    zone_metadata = {
        "method": "ndvi_percentiles",
        "n_classes": int(n_classes),
        "thresholds": thresholds_info,
        "months_used": used_months,
        "months_skipped": skipped_months,
        "stability": stability_info,
        "palette": palette,
    }

    zone_clean = ee.Image(zone_clean).setMulti(zone_metadata)

    return ZoneArtifacts(
        zone_image=zone_clean,
        zone_vectors=zone_vectors,
        zonal_stats=zonal_stats,
        geometry=geometry,
    )


def _sanitize_name(name: str) -> str:
    if not name:
        return "aoi"
    cleaned = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "aoi"


def export_prefix(aoi_name: str, months: Sequence[str]) -> str:
    return _legacy_zones.export_prefix(aoi_name, months)


def _export_image(
    image: ee.Image,
    geometry: ee.Geometry,
    *,
    destination: str,
    prefix: str,
    filename: str,
    drive_folder: str | None,
    gcs_bucket: str | None,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "image": image,
        "description": f"{prefix}_zone_raster",
        "region": geometry,
        "scale": SCALE,
        "maxPixels": 5_000_000_000,
        "fileFormat": "GeoTIFF",
        "crs": EXPORT_CRS,
        "nodata": 0,
    }

    if destination == "drive":
        task = ee.batch.Export.image.toDrive(
            fileNamePrefix=filename,
            folder=drive_folder,
            **params,
        )
        path = (
            f"drive://{drive_folder}/{filename}" if drive_folder else f"drive://{filename}"
        )
    elif destination == "gcs":
        if not gcs_bucket:
            raise ValueError("gcs_bucket must be provided for Cloud Storage exports.")
        task = ee.batch.Export.image.toCloudStorage(
            bucket=gcs_bucket,
            fileNamePrefix=filename,
            **params,
        )
        path = f"gs://{gcs_bucket}/{filename}"
    else:
        raise ValueError("destination must be either 'drive' or 'gcs'.")

    task.start()
    return {"task": task, "path": path}


def _export_table(
    collection: ee.FeatureCollection,
    *,
    destination: str,
    prefix: str,
    filename: str,
    drive_folder: str | None,
    gcs_bucket: str | None,
    file_format: str,
    selectors: Sequence[str] | None = None,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "collection": collection,
        "description": f"{prefix}_{file_format.lower()}",
        "fileFormat": file_format,
    }
    if selectors is not None:
        params["selectors"] = list(selectors)

    if destination == "drive":
        task = ee.batch.Export.table.toDrive(
            fileNamePrefix=filename,
            folder=drive_folder,
            **params,
        )
        path = (
            f"drive://{drive_folder}/{filename}" if drive_folder else f"drive://{filename}"
        )
    elif destination == "gcs":
        if not gcs_bucket:
            raise ValueError("gcs_bucket must be provided for Cloud Storage exports.")
        task = ee.batch.Export.table.toCloudStorage(
            bucket=gcs_bucket,
            fileNamePrefix=filename,
            **params,
        )
        path = f"gs://{gcs_bucket}/{filename}"
    else:
        raise ValueError("destination must be either 'drive' or 'gcs'.")

    task.start()
    return {"task": task, "path": path}


def export_selected_period_zones(
    aoi_geojson_or_geom: Union[dict, ee.Geometry] | None = None,
    *,
    months: Sequence[str],
    aoi_name: str,
    aoi_geojson: Union[dict, ee.Geometry] | None = None,
    destination: str = "drive",
    drive_folder: str | None = None,
    gcs_bucket: str | None = None,
    gcs_prefix: str | None = None,
    raster_filename: str | None = None,
    vector_filename: str | None = None,
    stats_filename: str | None = None,
    include_stats: bool = True,
    geometry: ee.Geometry | None = None,
    **build_kwargs: Any,
) -> Dict[str, Any]:
    if aoi_geojson_or_geom is None:
        if aoi_geojson is not None:
            aoi_geojson_or_geom = aoi_geojson
        elif "aoi_geojson" in build_kwargs:
            aoi_geojson_or_geom = build_kwargs.pop("aoi_geojson")

    legacy_destination = build_kwargs.pop("export_target", None)
    destination_value = destination
    if legacy_destination is not None and (
        not destination_value or destination_value == "drive"
    ):
        destination_value = str(legacy_destination)

    if "mmu_ha" in build_kwargs and "min_mapping_unit_ha" not in build_kwargs:
        build_kwargs["min_mapping_unit_ha"] = build_kwargs.pop("mmu_ha")

    if "include_zonal_stats" in build_kwargs:
        include_stats = bool(build_kwargs.pop("include_zonal_stats"))

    if "simplify_tol_m" in build_kwargs and "simplify_tolerance_m" not in build_kwargs:
        build_kwargs["simplify_tolerance_m"] = build_kwargs.pop("simplify_tol_m")

    if gcs_prefix is None and "gcs_prefix" in build_kwargs:
        gcs_prefix = build_kwargs.pop("gcs_prefix")

    if geometry is None and "geometry" in build_kwargs:
        geometry = build_kwargs.pop("geometry")
    else:
        build_kwargs.pop("geometry", None)

    build_kwargs.pop("start_date", None)
    build_kwargs.pop("end_date", None)

    build_include_stats = bool(build_kwargs.pop("include_stats", include_stats))
    build_kwargs["include_stats"] = build_include_stats

    destination = (destination_value or "drive").strip().lower()
    if destination not in {"drive", "gcs", "zip"}:
        raise ValueError("destination must be one of drive, gcs, or zip")

    cleaned_prefix: Optional[str] = None
    if gcs_prefix is not None:
        trimmed = gcs_prefix.strip().strip("/")
        cleaned_prefix = trimmed or None

    if aoi_geojson_or_geom is None:
        raise TypeError("aoi_geojson_or_geom must be provided")

    aoi_input: Union[dict, ee.Geometry, None] = aoi_geojson_or_geom
    if aoi_input is None:
        aoi_input = geometry

    artifacts = build_zone_artifacts(
        aoi_input,
        months=months,
        **build_kwargs,
    )

    export_geometry = geometry or artifacts.geometry
    prefix_root = export_prefix(aoi_name, months)
    raster_base = raster_filename or f"{prefix_root}_zones"
    vector_base = vector_filename or f"{prefix_root}_zones"
    stats_base = stats_filename or f"{prefix_root}_stats"

    def _apply_gcs_prefix(name: str) -> str:
        if not cleaned_prefix:
            return name
        return f"{cleaned_prefix}/{name}"

    tasks: Dict[str, Any] = {}
    paths: Dict[str, Any] = {}

    if destination == "zip":
        vector_components = {
            "shp": f"{vector_base}.shp",
            "dbf": f"{vector_base}.dbf",
            "shx": f"{vector_base}.shx",
            "prj": f"{vector_base}.prj",
        }
        paths = {
            "raster": f"{raster_base}.tif",
            "vectors": vector_components["shp"],
            "vector_components": vector_components,
            "zonal_stats": (
                f"{stats_base}.csv"
                if build_include_stats and artifacts.zonal_stats is not None
                else None
            ),
        }
    else:
        raster_target = raster_base
        vector_target = vector_base
        stats_target = stats_base
        if destination == "gcs":
            if not gcs_bucket:
                raise ValueError(
                    "gcs_bucket must be provided for Cloud Storage exports."
                )
            raster_target = _apply_gcs_prefix(raster_base)
            vector_target = _apply_gcs_prefix(vector_base)
            stats_target = _apply_gcs_prefix(stats_base)

        image_task = _export_image(
            artifacts.zone_image,
            export_geometry,
            destination=destination,
            prefix=prefix_root,
            filename=raster_target,
            drive_folder=drive_folder,
            gcs_bucket=gcs_bucket,
        )

        vector_task = _export_table(
            artifacts.zone_vectors,
            destination=destination,
            prefix=prefix_root,
            filename=vector_target,
            drive_folder=drive_folder,
            gcs_bucket=gcs_bucket,
            file_format="SHP",
            selectors=["zone", "zone_id", "area_m2", "area_ha"],
        )

        stats_task = None
        if build_include_stats and artifacts.zonal_stats is not None:
            stats_task = _export_table(
                artifacts.zonal_stats,
                destination=destination,
                prefix=prefix_root,
                filename=stats_target,
                drive_folder=drive_folder,
                gcs_bucket=gcs_bucket,
                file_format="CSV",
                selectors=[
                    "zone",
                    "zone_id",
                    "area_m2",
                    "area_ha",
                    "NDVI_mean",
                    "NDVI_median",
                    "NDVI_stdDev",
                    "NDVI_cv",
                ],
            )

        tasks = {
            "raster": image_task,
            "vector": vector_task,
            "stats": stats_task,
        }

        if destination == "gcs":
            vector_components = {
                "shp": f"gs://{gcs_bucket}/{vector_target}.shp",
                "dbf": f"gs://{gcs_bucket}/{vector_target}.dbf",
                "shx": f"gs://{gcs_bucket}/{vector_target}.shx",
                "prj": f"gs://{gcs_bucket}/{vector_target}.prj",
            }
            paths = {
                "raster": f"gs://{gcs_bucket}/{raster_target}.tif",
                "vectors": vector_components["shp"],
                "vector_components": vector_components,
                "zonal_stats": (
                    f"gs://{gcs_bucket}/{stats_target}.csv"
                    if build_include_stats and artifacts.zonal_stats is not None
                    else None
                ),
            }
        else:
            paths = {
                "raster": image_task["path"],
                "vectors": vector_task["path"],
                "vector_components": None,
                "zonal_stats": stats_task["path"] if stats_task else None,
            }

    def _get_property(name: str, default: Any = None) -> Any:
        value = artifacts.zone_image.get(name)
        if value is None:
            return default
        info = value.getInfo()
        return default if info is None else info

    used_months = list(_get_property("months_used", list(months)))
    skipped_months = list(_get_property("months_skipped", []))
    thresholds = list(_get_property("thresholds", []))
    palette = list(_get_property("palette", []))
    stability = _get_property("stability", {}) or {}

    metadata = {
        "months_used": used_months,
        "used_months": used_months,
        "months_skipped": skipped_months,
        "skipped_months": skipped_months,
        "thresholds": thresholds,
        "percentile_thresholds": thresholds,
        "palette": palette,
        "stability": stability,
    }

    prefix_value = (
        _apply_gcs_prefix(raster_base) if destination == "gcs" else raster_base
    )

    result: Dict[str, Any] = {
        "prefix": prefix_value,
        "destination": destination,
        "paths": paths,
        "tasks": tasks,
        "metadata": metadata,
        "artifacts": artifacts,
    }

    if palette:
        result["palette"] = palette
    if thresholds:
        result["thresholds"] = thresholds

    if destination == "gcs":
        result["bucket"] = gcs_bucket
        if cleaned_prefix:
            result["gcs_prefix"] = cleaned_prefix
    elif destination == "drive" and drive_folder:
        result["folder"] = drive_folder

    return result


_SYNC_NAMES = [
    "_percentile_thresholds",
    "_classify_by_percentiles",
    "_build_percentile_zones",
    "_build_composite_series",
    "_compute_ndvi",
    "_ndvi_temporal_stats",
    "_stability_mask",
    "_prepare_vectors",
    "_apply_cleanup",
    "_simplify_vectors",
    "_normalise_feature",
    "area_ha",
    "_ordered_months",
    "_connected_component_area",
    "_pixel_count",
    "_months_from_dates",
]


def _legacy_wrapper(name: str):
    func = getattr(_legacy_zones, name)

    def _wrapped(*args, **kwargs):
        _legacy_zones.ee = ee
        _legacy_zones.gee = gee
        for attr in _SYNC_NAMES:
            if attr in globals():
                setattr(_legacy_zones, attr, globals()[attr])
        return func(*args, **kwargs)

    return _wrapped


_percentile_thresholds = _legacy_wrapper("_percentile_thresholds")
_classify_by_percentiles = _legacy_wrapper("_classify_by_percentiles")
_build_percentile_zones = _legacy_wrapper("_build_percentile_zones")
_build_composite_series = _legacy_wrapper("_build_composite_series")
_compute_ndvi = _legacy_wrapper("_compute_ndvi")
_ndvi_temporal_stats = _legacy_wrapper("_ndvi_temporal_stats")
_stability_mask = _legacy_wrapper("_stability_mask")
_prepare_vectors = _legacy_wrapper("_prepare_vectors")
_apply_cleanup = _legacy_wrapper("_apply_cleanup")
_normalise_feature = _legacy_wrapper("_normalise_feature")
_simplify_vectors = _legacy_wrapper("_simplify_vectors")
area_ha = _legacy_wrapper("area_ha")
_ordered_months = _legacy_wrapper("_ordered_months")
_connected_component_area = _legacy_wrapper("_connected_component_area")
_pixel_count = _legacy_wrapper("_pixel_count")
_months_from_dates = _legacy_wrapper("_months_from_dates")


def _prepare_selected_period_artifacts(*args, **kwargs):
    _legacy_zones.ee = ee
    _legacy_zones.gee = gee
    for attr in _SYNC_NAMES:
        if attr in globals():
            setattr(_legacy_zones, attr, globals()[attr])
    artifacts, metadata = _legacy_zones._prepare_selected_period_artifacts(*args, **kwargs)
    if isinstance(metadata, dict):
        palette = metadata.get("palette")
        if palette:
            count = max(1, len(palette))
            metadata["palette"] = list(ZONE_PALETTE[:count])
    return artifacts, metadata
resolve_export_bucket = _legacy_zones.resolve_export_bucket
STABILITY_THRESHOLD_SEQUENCE = _legacy_zones.STABILITY_THRESHOLD_SEQUENCE
STABILITY_MASK_EMPTY_ERROR = _legacy_zones.STABILITY_MASK_EMPTY_ERROR
NDVI_MASK_EMPTY_ERROR = _legacy_zones.NDVI_MASK_EMPTY_ERROR


