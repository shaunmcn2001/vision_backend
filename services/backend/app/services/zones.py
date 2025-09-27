"""Production zone workflow built on Sentinel-2 monthly composites."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import os
from typing import Dict, List, Mapping, Sequence, Union

import ee

from app import gee
from app.exports import sanitize_name


DEFAULT_CLOUD_PROB_MAX = 40
DEFAULT_N_CLASSES = 5
DEFAULT_CV_THRESHOLD = 0.25
DEFAULT_MIN_MAPPING_UNIT_HA = 0.5
DEFAULT_SMOOTH_KERNEL_PX = 1
DEFAULT_SIMPLIFY_TOL_M = 5
DEFAULT_METHOD = "ndvi_percentiles"
DEFAULT_SAMPLE_SIZE = 8000
DEFAULT_SCALE = 10
DEFAULT_CRS = "EPSG:4326"


def _subtract_years(reference: date, years: int) -> date:
    """Return ``reference`` shifted ``years`` into the past."""

    try:
        return reference.replace(year=reference.year - years)
    except ValueError:
        # Handle 29 February in leap years gracefully.
        return reference.replace(month=2, day=28, year=reference.year - years)


def _month_sequence(start: date, end: date) -> List[str]:
    """Generate ``YYYY-MM`` strings spanning ``start`` through ``end`` inclusive."""

    if start > end:
        raise ValueError("start date must not be after end date")

    months: List[str] = []
    current = date(start.year, start.month, 1)
    terminal = date(end.year, end.month, 1)
    while current <= terminal:
        months.append(current.strftime("%Y-%m"))
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)
    return months


@dataclass(frozen=True)
class ZoneArtifacts:
    """Container for the images/vectors used for production zone exports."""

    zone_image: ee.Image
    zone_vectors: ee.FeatureCollection
    zonal_stats: ee.FeatureCollection | None
    geometry: ee.Geometry


@dataclass(frozen=True)
class ProductionWindow:
    """Metadata describing a rolling multi-year production window."""

    start_month: str
    end_month: str
    months: List[str]


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


def _normalise_growth_months(growth_months: Sequence[str] | None) -> List[str]:
    if not growth_months:
        return []

    normalised: List[str] = []
    for value in growth_months:
        month_str = str(value).strip()
        if len(month_str) != 2 or not month_str.isdigit():
            raise ValueError("growth months must be MM strings between 01 and 12")
        if int(month_str) < 1 or int(month_str) > 12:
            raise ValueError("growth months must be MM strings between 01 and 12")
        if month_str not in normalised:
            normalised.append(month_str)
    return normalised


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


def production5y_export_prefix(aoi_name: str, start_month: str, end_month: str) -> str:
    """Return the export prefix for a rolling 5-year production window."""

    safe_name = sanitize_name(aoi_name or "aoi")
    start_fmt = start_month.replace("-", "")
    end_fmt = end_month.replace("-", "")
    return f"zones/PROD5Y_{start_fmt}_{end_fmt}_{safe_name}_zones"


def month_bounds(months: Sequence[str]) -> tuple[str, str]:
    """Return the first and last months used in an export request."""

    return _month_bounds(months)


def resolve_export_bucket(explicit: str | None = None) -> str:
    bucket = (explicit or os.getenv("GEE_GCS_BUCKET") or os.getenv("GCS_BUCKET") or "").strip()
    if not bucket:
        raise RuntimeError("GEE_GCS_BUCKET or GCS_BUCKET must be set for zone exports")
    return bucket


def _resolve_geometry(aoi: Union[dict, ee.Geometry]) -> ee.Geometry:
    if isinstance(aoi, ee.Geometry):
        return aoi
    return gee.geometry_from_geojson(aoi)


def _build_monthly_composites(
    geometry: ee.Geometry, months: Sequence[str], cloud_prob_max: int
) -> List[ee.Image]:
    composites: List[ee.Image] = []
    for month in _ordered_months(months):
        _, composite = gee.monthly_sentinel2_collection(geometry, month, cloud_prob_max)
        composites.append(
            composite.resample("bilinear").reproject(DEFAULT_CRS, None, DEFAULT_SCALE)
        )
    return composites


def _compute_ndvi(image: ee.Image) -> ee.Image:
    return image.normalizedDifference(["B8", "B4"]).rename("NDVI")


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
    collection = ee.ImageCollection([img.rename("NDVI") for img in images])
    mean = collection.mean().rename("NDVI_mean")
    median = collection.median().rename("NDVI_median")
    std = collection.reduce(ee.Reducer.stdDev()).rename("NDVI_stdDev")
    return {"mean": mean, "median": median, "std": std}


def _ndvi_cv(mean_image: ee.Image, std_image: ee.Image) -> ee.Image:
    mean_abs = mean_image.abs()
    safe_mean = mean_abs.where(mean_abs.gte(1e-6), ee.Image.constant(1))
    cv = std_image.divide(safe_mean)
    cv = cv.where(mean_abs.lt(1e-6), 1)
    return cv.rename("NDVI_cv")


def _stability_mask(cv_image: ee.Image, threshold: float) -> ee.Image:
    return cv_image.lte(threshold)


def _percentile_thresholds(
    image: ee.Image, geometry: ee.Geometry, n_classes: int
) -> ee.List:
    percent_steps = ee.List.sequence(1, n_classes - 1)
    percentiles = percent_steps.map(lambda i: ee.Number(i).multiply(100).divide(n_classes))
    output_names = percent_steps.map(
        lambda i: ee.String("cut_").cat(ee.Number(i).format("%02d"))
    )
    reducer = ee.Reducer.percentile(percentiles, output_names)
    stats = image.reduceRegion(
        reducer=reducer,
        geometry=geometry,
        scale=DEFAULT_SCALE,
        bestEffort=True,
        tileScale=4,
        maxPixels=gee.MAX_PIXELS,
    )
    return output_names.map(lambda name: ee.Number(stats.get(name)))


def _classify_by_percentiles(
    image: ee.Image, geometry: ee.Geometry, n_classes: int
) -> ee.Image:
    band_name = ee.String(image.bandNames().get(0))
    image = image.rename(band_name)
    thresholds = _percentile_thresholds(image, geometry, n_classes)
    initial = ee.Image.constant(n_classes)

    def _assign(current, item):
        current_image = ee.Image(current)
        info = ee.List(item)
        idx = ee.Number(info.get(0))
        threshold = ee.Number(info.get(1))
        class_value = idx.add(1)
        return current_image.where(image.lte(threshold), class_value)

    pairs = ee.List.sequence(0, thresholds.size().subtract(1)).zip(thresholds)
    classified = ee.Image(pairs.iterate(_assign, initial))
    return classified.rename("zone")


def _connected_component_area(classified: ee.Image, n_classes: int) -> ee.Image:
    pixel_area = ee.Image.pixelArea()
    area_image = ee.Image.constant(0)
    for class_id in range(1, n_classes + 1):
        mask = classified.eq(class_id)
        counts = mask.connectedPixelCount(maxSize=1_000_000, eightConnected=True)
        area = counts.multiply(pixel_area)
        area_image = area_image.where(mask, area)
    return area_image.rename("component_area")


def _apply_cleanup(
    classified: ee.Image,
    geometry: ee.Geometry,
    *,
    n_classes: int,
    smooth_kernel_px: int,
    min_mapping_unit_ha: float,
) -> ee.Image:
    smooth_radius = max(1, int(smooth_kernel_px))
    smoothed = classified.focal_mode(radius=smooth_radius, units="pixels", iterations=1)
    component_area = _connected_component_area(smoothed, n_classes)
    min_area_m2 = max(min_mapping_unit_ha, 0) * 10_000
    majority_large = smoothed.focal_mode(radius=smooth_radius + 1, units="pixels", iterations=1)
    small_mask = component_area.lt(min_area_m2)
    cleaned = smoothed.where(small_mask, majority_large)
    cleaned = cleaned.clip(geometry)

    mask = cleaned.mask()
    closed_mask = mask.focal_max(radius=1, units="pixels", iterations=1).focal_min(
        radius=1, units="pixels", iterations=1
    )
    gap_mask = closed_mask.And(mask.Not())
    filler = cleaned.focal_mode(radius=1, units="pixels", iterations=1)
    cleaned = cleaned.where(gap_mask, filler)
    cleaned = cleaned.updateMask(closed_mask)
    return cleaned


def _simplify_vectors(vectors: ee.FeatureCollection, tolerance_m: float) -> ee.FeatureCollection:
    if tolerance_m <= 0:
        return vectors

    def _simplify(feature: ee.Feature) -> ee.Feature:
        geom = feature.geometry().simplify(maxError=tolerance_m, preserveTopology=True)
        zone_value = ee.Number(feature.get("zone")).toInt()
        return feature.setGeometry(geom).set({"zone": zone_value, "zone_id": zone_value})

    return ee.FeatureCollection(vectors.map(_simplify))


def _prepare_vectors(
    zone_image: ee.Image,
    geometry: ee.Geometry,
    *,
    tolerance_m: float,
) -> ee.FeatureCollection:
    vectors = zone_image.reduceToVectors(
        geometry=geometry,
        scale=DEFAULT_SCALE,
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

    return _simplify_vectors(vectors, tolerance_m)


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
    area_ha = geometry.area(maxError=1).divide(10_000)
    ordered = [stats_images[name].rename(name) for name in sorted(stats_images)]
    stats_image = ee.Image.cat(ordered)
    stats = stats_image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=DEFAULT_SCALE,
        bestEffort=True,
        tileScale=4,
        maxPixels=gee.MAX_PIXELS,
    )
    zone_value = ee.Number(feature.get("zone")).toInt()
    return feature.set(stats).set({"area_ha": area_ha, "zone": zone_value, "zone_id": zone_value})


def _build_percentile_zones(
    *,
    ndvi_stats: Mapping[str, ee.Image],
    geometry: ee.Geometry,
    n_classes: int,
    smooth_kernel_px: int,
    min_mapping_unit_ha: float,
) -> ee.Image:
    ranked = _classify_by_percentiles(
        ndvi_stats["mean"].updateMask(ndvi_stats["stability"]), geometry, n_classes
    )
    ranked = ranked.updateMask(ndvi_stats["stability"])
    cleaned = _apply_cleanup(
        ranked,
        geometry,
        n_classes=n_classes,
        smooth_kernel_px=smooth_kernel_px,
        min_mapping_unit_ha=min_mapping_unit_ha,
    )
    return cleaned.rename("zone")


def _normalise_feature(mean_image: ee.Image, geometry: ee.Geometry, name: str) -> ee.Image:
    band_name = ee.String(mean_image.bandNames().get(0))
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
    mean_value = ee.Number(stats.get(band_name, 0))
    std_value = ee.Number(std_stats.get(band_name, 0)).max(1e-6)
    return mean_image.subtract(mean_value).divide(ee.Image.constant(std_value)).rename(f"norm_{name}")


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


def _build_multiindex_zones(
    *,
    ndvi_stats: Mapping[str, ee.Image],
    composites: Sequence[ee.Image],
    geometry: ee.Geometry,
    n_classes: int,
    smooth_kernel_px: int,
    min_mapping_unit_ha: float,
    sample_size: int,
) -> tuple[ee.Image, Dict[str, ee.Image]]:
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
    cleaned = _apply_cleanup(
        ranked,
        geometry,
        n_classes=n_classes,
        smooth_kernel_px=smooth_kernel_px,
        min_mapping_unit_ha=min_mapping_unit_ha,
    )

    return cleaned.rename("zone"), mean_images


def _build_multiindex_zones_with_features(
    *,
    ndvi_stats: Mapping[str, ee.Image],
    feature_images: Mapping[str, ee.Image],
    geometry: ee.Geometry,
    n_classes: int,
    smooth_kernel_px: int,
    min_mapping_unit_ha: float,
    sample_size: int,
) -> tuple[ee.Image, Dict[str, ee.Image]]:
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
    cleaned = _apply_cleanup(
        ranked,
        geometry,
        n_classes=n_classes,
        smooth_kernel_px=smooth_kernel_px,
        min_mapping_unit_ha=min_mapping_unit_ha,
    )

    return cleaned.rename("zone"), masked_features


def build_zone_artifacts(
    aoi_geojson: Union[dict, ee.Geometry],
    *,
    months: Sequence[str],
    cloud_prob_max: int = DEFAULT_CLOUD_PROB_MAX,
    n_classes: int = DEFAULT_N_CLASSES,
    cv_mask_threshold: float = DEFAULT_CV_THRESHOLD,
    min_mapping_unit_ha: float = DEFAULT_MIN_MAPPING_UNIT_HA,
    smooth_kernel_px: int = DEFAULT_SMOOTH_KERNEL_PX,
    simplify_tolerance_m: float = DEFAULT_SIMPLIFY_TOL_M,
    method: str = DEFAULT_METHOD,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    include_stats: bool = True,
) -> ZoneArtifacts:
    if n_classes < 3 or n_classes > 7:
        raise ValueError("n_classes must be between 3 and 7")
    if min_mapping_unit_ha <= 0:
        raise ValueError("min_mapping_unit_ha must be positive")
    if smooth_kernel_px < 0:
        raise ValueError("smooth_kernel_px must be non-negative")
    if not months:
        raise ValueError("At least one month must be supplied")

    method_key = method.strip().lower()
    if method_key not in {"ndvi_percentiles", "multiindex_kmeans"}:
        raise ValueError("Unsupported method for production zones")

    gee.initialize()
    geometry = _resolve_geometry(aoi_geojson)

    composites = _build_monthly_composites(geometry, months, cloud_prob_max)
    if not composites:
        raise ValueError("No monthly composites were generated")

    ndvi_images = [_compute_ndvi(image) for image in composites]
    stats = _ndvi_temporal_stats(ndvi_images)
    cv_image = _ndvi_cv(stats["mean"], stats["std"])
    stability = _stability_mask(cv_image, cv_mask_threshold)
    stats = {**stats, "cv": cv_image, "stability": stability}

    if method_key == "ndvi_percentiles":
        zone_image = _build_percentile_zones(
            ndvi_stats=stats,
            geometry=geometry,
            n_classes=n_classes,
            smooth_kernel_px=smooth_kernel_px,
            min_mapping_unit_ha=min_mapping_unit_ha,
        )
        extra_means: Dict[str, ee.Image] = {}
    else:
        zone_image, mean_images = _build_multiindex_zones(
            ndvi_stats=stats,
            composites=composites,
            geometry=geometry,
            n_classes=n_classes,
            smooth_kernel_px=smooth_kernel_px,
            min_mapping_unit_ha=min_mapping_unit_ha,
            sample_size=sample_size,
        )
        extra_means = {name: image.updateMask(stats["stability"]) for name, image in mean_images.items()}

    zone_image = zone_image.updateMask(zone_image.neq(0)).toInt16()
    zone_image = zone_image.rename("zone").reproject(DEFAULT_CRS, None, DEFAULT_SCALE)

    vectors = _prepare_vectors(zone_image, geometry, tolerance_m=simplify_tolerance_m)

    stats_collection = None
    if include_stats:
        stats_images = _collect_stats_images(stats, extra_means)
        stats_collection = ee.FeatureCollection(
            vectors.map(lambda feature: _add_zonal_stats(feature, stats_images))
        )

    return ZoneArtifacts(
        zone_image=zone_image,
        zone_vectors=vectors,
        zonal_stats=stats_collection,
        geometry=geometry,
    )


def build_production5y_zone_artifacts(
    aoi_geojson: Union[dict, ee.Geometry],
    *,
    years_back: int = 5,
    growth_months: Sequence[str] | None = None,
    cloud_prob_max: int = DEFAULT_CLOUD_PROB_MAX,
    n_classes: int = DEFAULT_N_CLASSES,
    cv_mask_threshold: float = DEFAULT_CV_THRESHOLD,
    min_mapping_unit_ha: float = DEFAULT_MIN_MAPPING_UNIT_HA,
    smooth_kernel_px: int = DEFAULT_SMOOTH_KERNEL_PX,
    simplify_tolerance_m: float = DEFAULT_SIMPLIFY_TOL_M,
    method: str = DEFAULT_METHOD,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    dem_asset: str | None = None,
    include_stats: bool = True,
) -> tuple[ZoneArtifacts, ProductionWindow]:
    if years_back <= 0:
        raise ValueError("years_back must be positive")
    if n_classes < 3 or n_classes > 7:
        raise ValueError("n_classes must be between 3 and 7")
    if min_mapping_unit_ha <= 0:
        raise ValueError("min_mapping_unit_ha must be positive")
    if smooth_kernel_px < 0:
        raise ValueError("smooth_kernel_px must be non-negative")

    method_key = method.strip().lower()
    if method_key not in {"ndvi_percentiles", "multiindex_kmeans"}:
        raise ValueError("Unsupported method for production zones")

    growth_filters = _normalise_growth_months(growth_months)

    gee.initialize()
    geometry = _resolve_geometry(aoi_geojson)

    end_date = datetime.utcnow().date()
    start_date = _subtract_years(end_date, years_back)
    months = _month_sequence(start_date, end_date)
    if not months:
        raise ValueError("No months available for the requested window")

    composites: List[tuple[str, ee.Image]] = []
    for month in months:
        _, composite = gee.monthly_sentinel2_collection(geometry, month, cloud_prob_max)
        composite = (
            composite.resample("bilinear").reproject(DEFAULT_CRS, None, DEFAULT_SCALE)
        )
        composites.append((month, composite))

    ndvi_by_month: Dict[str, ee.Image] = {}
    ndvi_images: List[ee.Image] = []
    ndre_images: List[ee.Image] = []
    ndmi_images: List[ee.Image] = []

    for month, image in composites:
        ndvi = _compute_ndvi(image)
        ndvi_by_month[month] = ndvi
        month_code = month[5:7]
        if not growth_filters or month_code in growth_filters:
            ndvi_images.append(ndvi)
            ndre_images.append(_compute_ndre(image))
            ndmi_images.append(_compute_ndmi(image))

    if not ndvi_images:
        raise ValueError("No monthly composites matched the growth_months filter")

    stats = _ndvi_temporal_stats(ndvi_images)
    cv_image = _ndvi_cv(stats["mean"], stats["std"])
    stability = _stability_mask(cv_image, cv_mask_threshold)
    stats = {**stats, "cv": cv_image, "stability": stability}

    bsi_images: List[ee.Image] = []
    for month, image in composites:
        bsi = _compute_bsi(image)
        ndvi = ndvi_by_month[month]
        month_code = month[5:7]
        if growth_filters and month_code not in growth_filters:
            mask = ee.Image.constant(1)
        else:
            mask = ndvi.lt(0.2)
        bsi_images.append(bsi.updateMask(mask))

    bsi_collection = ee.ImageCollection(bsi_images)
    bsi_ref = bsi_collection.median().rename("BSI_ref")
    bsi_ref = bsi_ref.reproject(DEFAULT_CRS, None, DEFAULT_SCALE).clip(geometry)

    dem_id = (dem_asset or "USGS/SRTMGL1_003").strip()
    dem_image = ee.Image(dem_id).resample("bilinear").reproject(
        DEFAULT_CRS, None, DEFAULT_SCALE
    )
    dem_image = dem_image.rename("elevation").clip(geometry)
    slope_image = ee.Terrain.slope(dem_image).rename("slope").reproject(
        DEFAULT_CRS, None, DEFAULT_SCALE
    ).clip(geometry)

    if method_key == "ndvi_percentiles":
        zone_image = _build_percentile_zones(
            ndvi_stats=stats,
            geometry=geometry,
            n_classes=n_classes,
            smooth_kernel_px=smooth_kernel_px,
            min_mapping_unit_ha=min_mapping_unit_ha,
        )
        feature_means: Dict[str, ee.Image] = {
            "BSI_ref_mean": bsi_ref.updateMask(stats["stability"]),
        }
    else:
        ndre_collection = ee.ImageCollection(ndre_images)
        ndmi_collection = ee.ImageCollection(ndmi_images)
        ndre_mean = ndre_collection.mean().rename("NDRE_mean")
        ndmi_mean = ndmi_collection.mean().rename("NDMI_mean")
        feature_images = {
            "NDVI_mean": stats["mean"],
            "NDRE_mean": ndre_mean,
            "NDMI_mean": ndmi_mean,
            "BSI_ref_mean": bsi_ref,
            "elevation_mean": dem_image.rename("elevation_mean"),
            "slope_mean": slope_image.rename("slope_mean"),
        }
        zone_image, feature_means = _build_multiindex_zones_with_features(
            ndvi_stats=stats,
            feature_images=feature_images,
            geometry=geometry,
            n_classes=n_classes,
            smooth_kernel_px=smooth_kernel_px,
            min_mapping_unit_ha=min_mapping_unit_ha,
            sample_size=sample_size,
        )

    zone_image = zone_image.updateMask(zone_image.neq(0)).toInt16()
    zone_image = zone_image.rename("zone").reproject(DEFAULT_CRS, None, DEFAULT_SCALE)

    vectors = _prepare_vectors(zone_image, geometry, tolerance_m=simplify_tolerance_m)

    stats_collection = None
    if include_stats:
        stats_images = _collect_stats_images(stats, feature_means)
        stats_collection = ee.FeatureCollection(
            vectors.map(lambda feature: _add_zonal_stats(feature, stats_images))
        )

    window = ProductionWindow(start_month=months[0], end_month=months[-1], months=months)

    return (
        ZoneArtifacts(
            zone_image=zone_image,
            zone_vectors=vectors,
            zonal_stats=stats_collection,
            geometry=geometry,
        ),
        window,
    )


def start_zone_exports(
    artifacts: ZoneArtifacts,
    *,
    aoi_name: str,
    months: Sequence[str] | None,
    bucket: str,
    include_stats: bool = True,
    prefix_override: str | None = None,
) -> Dict[str, ee.batch.Task]:
    if prefix_override:
        prefix = prefix_override
    else:
        if months is None:
            raise ValueError("months must be provided when prefix_override is not set")
        prefix = _export_prefix(aoi_name, months)
    description_base = prefix.split("/")[-1][:90]

    raster_task = ee.batch.Export.image.toCloudStorage(
        image=artifacts.zone_image,
        description=f"{description_base}_raster",
        bucket=bucket,
        fileNamePrefix=prefix,
        region=artifacts.geometry,
        scale=DEFAULT_SCALE,
        crs=DEFAULT_CRS,
        maxPixels=gee.MAX_PIXELS,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": False, "noDataValue": 0},
    )
    raster_task.start()

    vector_task = ee.batch.Export.table.toCloudStorage(
        collection=artifacts.zone_vectors,
        description=f"{description_base}_vectors",
        bucket=bucket,
        fileNamePrefix=prefix,
        fileFormat="SHP",
    )
    vector_task.start()

    stats_task = None
    if include_stats and artifacts.zonal_stats is not None:
        stats_prefix = prefix + "_zonal_stats"
        stats_task = ee.batch.Export.table.toCloudStorage(
            collection=artifacts.zonal_stats,
            description=f"{description_base}_stats",
            bucket=bucket,
            fileNamePrefix=stats_prefix,
            fileFormat="CSV",
        )
        stats_task.start()

    return {"raster": raster_task, "vectors": vector_task, "stats": stats_task}


def start_zone_exports_drive(
    artifacts: ZoneArtifacts,
    *,
    folder: str,
    prefix: str,
    include_stats: bool = True,
) -> Dict[str, ee.batch.Task]:
    description_base = prefix.split("/")[-1][:90]

    raster_task = ee.batch.Export.image.toDrive(
        image=artifacts.zone_image,
        description=f"{description_base}_raster",
        folder=folder,
        fileNamePrefix=prefix,
        region=artifacts.geometry,
        scale=DEFAULT_SCALE,
        crs=DEFAULT_CRS,
        maxPixels=gee.MAX_PIXELS,
        fileFormat="GeoTIFF",
    )
    raster_task.start()

    vector_task = ee.batch.Export.table.toDrive(
        collection=artifacts.zone_vectors,
        description=f"{description_base}_vectors",
        folder=folder,
        fileNamePrefix=prefix,
        fileFormat="SHP",
    )
    vector_task.start()

    stats_task = None
    if include_stats and artifacts.zonal_stats is not None:
        stats_task = ee.batch.Export.table.toDrive(
            collection=artifacts.zonal_stats,
            description=f"{description_base}_stats",
            folder=folder,
            fileNamePrefix=f"{prefix}_zonal_stats",
            fileFormat="CSV",
        )
        stats_task.start()

    return {"raster": raster_task, "vectors": vector_task, "stats": stats_task}

