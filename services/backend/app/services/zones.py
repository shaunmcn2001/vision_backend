"""Sentinel-2 NDVI percentile zoning utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

import ee

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

    t = ee.String(ee.Algorithms.ObjectType(x))
    return ee.Number(
        ee.Algorithms.If(
            t.compareTo("Image").eq(0),
            ee.Error(
                ee.String("Expected Number for ")
                .cat(context)
                .cat("; got Image. Never use ee.Image.constant(...) with an Image."),
            ),
            x,
        )
    )


def _to_geometry(aoi_geojson_or_geom: Union[dict, ee.Geometry]) -> ee.Geometry:
    """Normalise GeoJSON/feature inputs to :class:`ee.Geometry`."""

    if isinstance(aoi_geojson_or_geom, ee.Geometry):
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
    cleaned = zones

    if smooth_radius_m > 0:
        cleaned = cleaned.focal_mode(
            radius=smooth_radius_m,
            kernelType="circle",
            units="meters",
        )

    if open_radius_m > 0:
        cleaned = cleaned.focal_min(
            radius=open_radius_m,
            kernelType="circle",
            units="meters",
        ).focal_max(
            radius=open_radius_m,
            kernelType="circle",
            units="meters",
        )

    if close_radius_m > 0:
        cleaned = cleaned.focal_max(
            radius=close_radius_m,
            kernelType="circle",
            units="meters",
        ).focal_min(
            radius=close_radius_m,
            kernelType="circle",
            units="meters",
        )

    cleaned = cleaned.updateMask(cleaned.gt(0)).rename("zone")

    pixel_area = ee.Image.pixelArea()
    min_area_m2 = float(min_mapping_unit_ha) * 10_000.0

    connected_area = pixel_area.addBands(cleaned).reduceConnectedComponents(
        reducer=ee.Reducer.sum(),
        labelBand="zone",
        maxSize=1 << 22,
    ).rename("component_area")

    small_components = connected_area.lt(min_area_m2)

    fill_radius = max(smooth_radius_m, open_radius_m, close_radius_m, 1)
    majority = cleaned.focal_mode(
        radius=fill_radius,
        kernelType="circle",
        units="meters",
    )

    filled = cleaned.where(small_components, majority)
    filled = filled.updateMask(filled.gt(0)).rename("zone")
    return filled.clip(geometry).toInt16()


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

    zone_clean = zone_clean.setMulti(zone_metadata)

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
    month_list = sorted({m: None for m in months}.keys())
    if not month_list:
        raise ValueError("Months are required to build an export prefix.")
    name = _sanitize_name(aoi_name)
    return f"{name}_{'_'.join(month_list)}"


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
    aoi_geojson_or_geom: Union[dict, ee.Geometry],
    *,
    months: Sequence[str],
    aoi_name: str,
    destination: str = "drive",
    drive_folder: str | None = None,
    gcs_bucket: str | None = None,
    raster_filename: str | None = None,
    vector_filename: str | None = None,
    stats_filename: str | None = None,
    include_stats: bool = True,
    **build_kwargs: Any,
) -> Dict[str, Any]:
    build_include_stats = bool(build_kwargs.get("include_stats", include_stats))
    build_kwargs = {**build_kwargs, "include_stats": build_include_stats}

    artifacts = build_zone_artifacts(
        aoi_geojson_or_geom,
        months=months,
        **build_kwargs,
    )

    prefix = export_prefix(aoi_name, months)
    raster_name = raster_filename or f"{prefix}_zones"
    vector_name = vector_filename or f"{prefix}_zones"
    stats_name = stats_filename or f"{prefix}_stats"

    image_task = _export_image(
        artifacts.zone_image,
        artifacts.geometry,
        destination=destination,
        prefix=prefix,
        filename=raster_name,
        drive_folder=drive_folder,
        gcs_bucket=gcs_bucket,
    )

    vector_task = _export_table(
        artifacts.zone_vectors,
        destination=destination,
        prefix=prefix,
        filename=vector_name,
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
            prefix=prefix,
            filename=stats_name,
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

    def _get_property(name: str, default: Any = None) -> Any:
        value = artifacts.zone_image.get(name)
        return value.getInfo() if value is not None else default

    metadata = {
        "months_used": _get_property("months_used", []),
        "months_skipped": _get_property("months_skipped", []),
        "thresholds": _get_property("thresholds", []),
        "palette": _get_property("palette", []),
        "stability": _get_property("stability", {}),
    }

    return {
        "prefix": prefix,
        "destination": destination,
        "tasks": {
            "raster": image_task,
            "vector": vector_task,
            "stats": stats_task,
        },
        "metadata": metadata,
    }


