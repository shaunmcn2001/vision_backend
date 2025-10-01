"""Production zone workflow built on Sentinel-2 monthly composites."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
import math
from typing import Dict, List, Mapping, Sequence, Tuple, Union

import ee

from app import gee
from app.exports import sanitize_name
from app.utils.geometry import area_ha


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

ZONE_PALETTE: tuple[str, ...] = (
    "#00441b",
    "#006d2c",
    "#238b45",
    "#41ae76",
    "#66c2a4",
    "#99d8c9",
    "#ccece6",
)

STABILITY_THRESHOLD_SEQUENCE = [0.5, 1.0, 1.5, 2.0]
MIN_STABILITY_SURVIVAL_RATIO = 0.2

STABILITY_MASK_EMPTY_ERROR = (
    "All pixels were masked out by the stability threshold when computing NDVI percentiles. "
    "Try lowering the coefficient of variation threshold, expanding the selected months, "
    "or switching the zone method."
)
@dataclass(frozen=True)
class ZoneArtifacts:
    """Container for the images/vectors used for production zone exports."""

    zone_image: ee.Image
    zone_vectors: ee.FeatureCollection
    zonal_stats: ee.FeatureCollection | None
    geometry: ee.Geometry


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
    if isinstance(aoi, ee.Geometry):
        return aoi
    return gee.geometry_from_geojson(aoi)


def _build_monthly_composites(
    geometry: ee.Geometry, months: Sequence[str], cloud_prob_max: int
) -> Tuple[List[tuple[str, ee.Image]], List[str]]:
    composites: List[tuple[str, ee.Image]] = []
    skipped: List[str] = []
    ordered = _ordered_months(months)
    for month in ordered:
        collection, composite = gee.monthly_sentinel2_collection(
            geometry, month, cloud_prob_max
        )
        try:
            scene_count = int(ee.Number(collection.size()).getInfo() or 0)
        except Exception as exc:  # pragma: no cover - server side failure
            raise RuntimeError(f"Failed to evaluate Sentinel-2 collection for {month}: {exc}")
        if scene_count == 0:
            skipped.append(month)
            continue

        reproj = (
            composite.resample("bilinear").reproject(DEFAULT_CRS, None, DEFAULT_SCALE)
        )
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
        except Exception as exc:  # pragma: no cover - server side failure
            raise RuntimeError(
                f"Failed to determine valid pixel count for {month}: {exc}"
            ) from exc

        if valid_pixels == 0:
            skipped.append(month)
            continue

        composites.append((month, reproj.clip(geometry)))

    return composites, skipped


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


def _pixel_count(image: ee.Image, geometry: ee.Geometry, *, context: str) -> int:
    try:
        reduced = image.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=geometry,
            scale=DEFAULT_SCALE,
            bestEffort=True,
            tileScale=4,
            maxPixels=gee.MAX_PIXELS,
        )
    except Exception as exc:  # pragma: no cover - server side failure
        raise RuntimeError(f"Failed to evaluate {context}: {exc}") from exc

    try:
        reduced_info = reduced.getInfo() if hasattr(reduced, "getInfo") else reduced
    except Exception as exc:  # pragma: no cover - server side failure
        raise RuntimeError(f"Failed to evaluate {context}: {exc}") from exc

    if isinstance(reduced_info, dict):
        values = list(reduced_info.values())
        value = values[0] if values else 0
    else:
        value = reduced_info

    try:
        numeric = float(value or 0)
    except Exception:  # pragma: no cover - guard for unexpected types
        numeric = 0.0

    if not math.isfinite(numeric):
        numeric = 0.0

    return int(max(numeric, 0))


def _percentile_thresholds(
    image: ee.Image, geometry: ee.Geometry, n_classes: int
) -> ee.List:
    percent_steps = list(range(1, n_classes))
    percentiles = ee.List([step * 100 / n_classes for step in percent_steps])
    output_names_py = [f"cut_{step:02d}" for step in percent_steps]
    output_names = ee.List(output_names_py)
    reducer = ee.Reducer.percentile(percentiles, output_names)
    stats = image.reduceRegion(
        reducer=reducer,
        geometry=geometry,
        scale=DEFAULT_SCALE,
        bestEffort=True,
        tileScale=4,
        maxPixels=gee.MAX_PIXELS,
    )

    try:
        keys_object = stats.keys() if hasattr(stats, "keys") else []
        if hasattr(keys_object, "getInfo"):
            keys_info = keys_object.getInfo()
        else:
            keys_info = list(keys_object)
    except Exception as exc:  # pragma: no cover - defensive guard for EE failures
        raise RuntimeError(f"Failed to evaluate NDVI percentile keys: {exc}") from exc

    present_keys = set(keys_info or [])
    missing_names = [name for name in output_names_py if name not in present_keys]
    if missing_names:
        raise ValueError(STABILITY_MASK_EMPTY_ERROR)

    return output_names.map(lambda name: ee.Number(stats.get(name)))


def _classify_by_percentiles(
    image: ee.Image, geometry: ee.Geometry, n_classes: int
) -> tuple[ee.Image, ee.List]:
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
    return classified.rename("zone"), thresholds


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
    base_radius_m = max(1, int(smooth_kernel_px)) * DEFAULT_SCALE

    smoothed = classified.focal_mode(radius=base_radius_m, units="meters", iterations=1)
    opened = smoothed.focal_min(radius=base_radius_m, units="meters", iterations=1).focal_max(
        radius=base_radius_m, units="meters", iterations=1
    )
    closed = opened.focal_max(radius=base_radius_m, units="meters", iterations=1).focal_min(
        radius=base_radius_m, units="meters", iterations=1
    )

    component_area = _connected_component_area(closed, n_classes)
    min_area_m2 = max(min_mapping_unit_ha, 0) * 10_000
    majority_large = closed.focal_mode(radius=base_radius_m, units="meters", iterations=1)
    small_mask = component_area.lt(min_area_m2)
    cleaned = closed.where(small_mask, majority_large)

    mask = cleaned.mask()
    closed_mask = mask.focal_max(radius=base_radius_m, units="meters", iterations=1).focal_min(
        radius=base_radius_m, units="meters", iterations=1
    )
    filler = cleaned.focal_mode(radius=base_radius_m, units="meters", iterations=1)
    cleaned = cleaned.where(mask.Not(), filler)
    cleaned = cleaned.updateMask(closed_mask)
    cleaned = cleaned.clip(geometry)
    return cleaned


def _simplify_vectors(vectors: ee.FeatureCollection, tolerance_m: float) -> ee.FeatureCollection:
    def _simplify(feature: ee.Feature) -> ee.Feature:
        geom = feature.geometry()
        if tolerance_m > 0:
            geom = geom.simplify(maxError=tolerance_m)
            geom = geom.buffer(0)
        zone_value = ee.Number(feature.get("zone")).toInt()
        area_m2 = geom.area(maxError=1)
        area_ha = area_m2.divide(10_000)
        return (
            feature.setGeometry(geom)
            .set({
                "zone": zone_value,
                "zone_id": zone_value,
                "area_m2": area_m2,
                "area_ha": area_ha,
            })
        )

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
) -> tuple[ee.Image, List[float]]:
    try:
        ranked, thresholds = _classify_by_percentiles(
            ndvi_stats["mean"].updateMask(ndvi_stats["stability"]), geometry, n_classes
        )
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    ranked = ranked.updateMask(ndvi_stats["stability"])

    try:
        percentile_thresholds: List[float] = [
            float(value) for value in (thresholds.getInfo() or [])
        ]
    except Exception as exc:  # pragma: no cover - server side failure
        raise RuntimeError(f"Failed to evaluate NDVI percentile thresholds: {exc}") from exc
    cleaned = _apply_cleanup(
        ranked,
        geometry,
        n_classes=n_classes,
        smooth_kernel_px=smooth_kernel_px,
        min_mapping_unit_ha=min_mapping_unit_ha,
    )
    return cleaned.rename("zone"), percentile_thresholds


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


def _prepare_selected_period_artifacts(
    aoi_geojson: Union[dict, ee.Geometry],
    *,
    geometry: ee.Geometry,
    months: Sequence[str],
    cloud_prob_max: int,
    n_classes: int,
    cv_mask_threshold: float,
    min_mapping_unit_ha: float,
    smooth_kernel_px: int,
    simplify_tol_m: float,
    method: str,
    sample_size: int,
    include_stats: bool,
) -> tuple[ZoneArtifacts, Dict[str, object]]:
    composites, skipped_months = _build_monthly_composites(
        geometry, months, cloud_prob_max
    )
    if not composites:
        raise ValueError("No valid Sentinel-2 scenes were found for the selected months")

    ordered_months = [month for month, _ in composites]
    ndvi_images = [_compute_ndvi(image) for _, image in composites]
    stats = _ndvi_temporal_stats(ndvi_images)
    cv_image = _ndvi_cv(stats["mean"], stats["std"])

    initial_threshold = float(cv_mask_threshold)
    thresholds_to_try: List[float] = [initial_threshold]
    for fallback in STABILITY_THRESHOLD_SEQUENCE:
        if fallback > initial_threshold + 1e-9:
            thresholds_to_try.append(fallback)

    total_pixels = _pixel_count(
        cv_image, geometry, context="stability baseline pixel count"
    )
    if total_pixels <= 0:
        raise ValueError(STABILITY_MASK_EMPTY_ERROR)

    stability = None
    final_threshold = initial_threshold
    survival_ratio = 0.0
    surviving_pixels = 0
    thresholds_tested: List[float] = []
    ratio_history: List[float] = []

    for idx, threshold in enumerate(thresholds_to_try):
        thresholds_tested.append(float(threshold))
        candidate = _stability_mask(cv_image, threshold)
        surviving_pixels = _pixel_count(
            candidate.updateMask(candidate),
            geometry,
            context="stability surviving pixel count",
        )
        ratio = (surviving_pixels / total_pixels) if total_pixels else 0.0
        ratio_history.append(ratio)
        stability = candidate
        final_threshold = float(threshold)
        survival_ratio = ratio
        if ratio >= MIN_STABILITY_SURVIVAL_RATIO or idx == len(thresholds_to_try) - 1:
            break

    if stability is None:
        raise ValueError(STABILITY_MASK_EMPTY_ERROR)

    low_confidence = final_threshold > initial_threshold + 1e-9

    stats = {**stats, "cv": cv_image, "stability": stability}

    mmu_value = max(min_mapping_unit_ha, 0)
    mmu_applied = True
    if isinstance(aoi_geojson, dict):
        try:
            if area_ha(aoi_geojson) < min_mapping_unit_ha:
                mmu_value = 0
                mmu_applied = False
        except Exception:  # pragma: no cover - fallback if area calc fails
            mmu_value = max(min_mapping_unit_ha, 0)
            mmu_applied = True

    method_key = method.strip().lower()
    extra_means: Dict[str, ee.Image] = {}

    percentile_thresholds: List[float] = []

    if method_key == "ndvi_percentiles":
        try:
            zone_image, percentile_thresholds = _build_percentile_zones(
                ndvi_stats=stats,
                geometry=geometry,
                n_classes=n_classes,
                smooth_kernel_px=smooth_kernel_px,
                min_mapping_unit_ha=mmu_value,
            )
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
    else:
        zone_image, mean_images = _build_multiindex_zones(
            ndvi_stats=stats,
            composites=[image for _, image in composites],
            geometry=geometry,
            n_classes=n_classes,
            smooth_kernel_px=smooth_kernel_px,
            min_mapping_unit_ha=mmu_value,
            sample_size=sample_size,
        )
        extra_means = {
            name: image.updateMask(stats["stability"])
            for name, image in mean_images.items()
        }

    zone_image = zone_image.updateMask(zone_image.neq(0)).toInt16()
    zone_image = zone_image.rename("zone").reproject(DEFAULT_CRS, None, DEFAULT_SCALE)

    vectors = _prepare_vectors(zone_image, geometry, tolerance_m=simplify_tol_m)

    stats_collection = None
    stats_images = _collect_stats_images(stats, extra_means)
    if include_stats:
        stats_collection = ee.FeatureCollection(
            vectors.map(lambda feature: _add_zonal_stats(feature, stats_images))
        )

    artifacts = ZoneArtifacts(
        zone_image=zone_image,
        zone_vectors=vectors,
        zonal_stats=stats_collection,
        geometry=geometry,
    )

    metadata: Dict[str, object] = {
        "used_months": ordered_months,
        "skipped_months": skipped_months,
        "mmu_applied": mmu_value > 0 and mmu_applied,
    }

    if percentile_thresholds:
        metadata["percentile_thresholds"] = percentile_thresholds
    metadata["palette"] = list(ZONE_PALETTE[: max(1, min(n_classes, len(ZONE_PALETTE)))])

    stability_metadata = {
        "initial_threshold": initial_threshold,
        "final_threshold": final_threshold,
        "survival_ratio": survival_ratio,
        "surviving_pixels": surviving_pixels,
        "total_pixels": total_pixels,
        "thresholds_tested": thresholds_tested,
        "low_confidence": low_confidence,
        "target_ratio": MIN_STABILITY_SURVIVAL_RATIO,
    }

    metadata["stability"] = stability_metadata
    metadata["low_confidence"] = low_confidence

    debug_block = metadata.setdefault("debug", {})
    debug_block["stability"] = {
        "initial_threshold": initial_threshold,
        "final_threshold": final_threshold,
        "thresholds_tested": thresholds_tested,
        "survival_ratios": ratio_history,
        "survival_ratio": survival_ratio,
        "surviving_pixels": surviving_pixels,
        "total_pixels": total_pixels,
        "target_ratio": MIN_STABILITY_SURVIVAL_RATIO,
        "low_confidence": low_confidence,
    }

    return artifacts, metadata


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

    artifacts, _metadata = _prepare_selected_period_artifacts(
        aoi_geojson,
        geometry=geometry,
        months=months,
        cloud_prob_max=cloud_prob_max,
        n_classes=n_classes,
        cv_mask_threshold=cv_mask_threshold,
        min_mapping_unit_ha=min_mapping_unit_ha,
        smooth_kernel_px=smooth_kernel_px,
        simplify_tol_m=simplify_tolerance_m,
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


def _task_payload(task: ee.batch.Task | None) -> Dict[str, object]:
    if task is None:
        return {}
    payload: Dict[str, object] = {"id": getattr(task, "id", None)}
    try:
        status = task.status() or {}
    except Exception:  # pragma: no cover - defensive
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
    aoi_geojson: Union[dict, ee.Geometry],
    aoi_name: str,
    months: Sequence[str],
    *,
    geometry: ee.Geometry | None = None,
    cloud_prob_max: int = DEFAULT_CLOUD_PROB_MAX,
    n_classes: int = DEFAULT_N_CLASSES,
    cv_mask_threshold: float = DEFAULT_CV_THRESHOLD,
    mmu_ha: float = DEFAULT_MIN_MAPPING_UNIT_HA,
    smooth_kernel_px: int = DEFAULT_SMOOTH_KERNEL_PX,
    simplify_tol_m: float = DEFAULT_SIMPLIFY_TOL_M,
    export_target: str = "zip",
    gcs_bucket: str | None = None,
    gcs_prefix: str | None = None,
    include_zonal_stats: bool = True,
) -> Dict[str, object]:
    if not months:
        raise ValueError("months must contain at least one YYYY-MM value")
    ordered_months = _ordered_months(months)
    if n_classes < 3 or n_classes > 7:
        raise ValueError("n_classes must be between 3 and 7")
    if smooth_kernel_px < 0:
        raise ValueError("smooth_kernel_px must be non-negative")
    if mmu_ha < 0:
        raise ValueError("mmu_ha must be non-negative")

    gee.initialize()
    geometry = geometry or _resolve_geometry(aoi_geojson)

    artifacts, metadata = _prepare_selected_period_artifacts(
        aoi_geojson,
        geometry=geometry,
        months=ordered_months,
        cloud_prob_max=cloud_prob_max,
        n_classes=n_classes,
        cv_mask_threshold=cv_mask_threshold,
        min_mapping_unit_ha=mmu_ha,
        smooth_kernel_px=smooth_kernel_px,
        simplify_tol_m=simplify_tol_m,
        method=DEFAULT_METHOD,
        sample_size=DEFAULT_SAMPLE_SIZE,
        include_stats=include_zonal_stats,
    )

    metadata = dict(metadata)
    used_months: List[str] = list(metadata.get("used_months", []))
    skipped: List[str] = list(metadata.get("skipped_months", []))
    if not used_months:
        raise ValueError("No valid Sentinel-2 scenes available for the selected period")

    prefix_base = export_prefix(aoi_name, used_months)
    stats_name = f"{prefix_base}_zonal_stats.csv" if include_zonal_stats else None

    export_target = (export_target or "zip").strip().lower()
    if export_target not in {"zip", "gcs", "drive"}:
        raise ValueError("export_target must be one of zip, gcs, or drive")

    vector_components = {
        "shp": f"{prefix_base}.shp",
        "dbf": f"{prefix_base}.dbf",
        "shx": f"{prefix_base}.shx",
        "prj": f"{prefix_base}.prj",
    }

    metadata.update(
        {
            "used_months": used_months,
            "skipped_months": skipped,
            "mmu_applied": bool(metadata.get("mmu_applied", True)),
        }
    )

    palette = metadata.get("palette") if isinstance(metadata, dict) else None
    thresholds = metadata.get("percentile_thresholds") if isinstance(metadata, dict) else None

    result: Dict[str, object] = {
        "paths": {
            "raster": f"{prefix_base}.tif",
            "vectors": vector_components["shp"],
            "vector_components": vector_components,
            "zonal_stats": stats_name,
        },
        "tasks": {},
        "prefix": prefix_base,
        "metadata": metadata,
        "artifacts": artifacts,
    }

    if palette:
        result["palette"] = palette
    if thresholds:
        result["thresholds"] = thresholds

    debug_info = metadata.get("debug") if isinstance(metadata, dict) else None
    if debug_info:
        result["debug"] = debug_info

    if export_target == "zip":
        return result

    if export_target == "gcs":
        bucket = resolve_export_bucket(gcs_bucket)
        cleaned_prefix = prefix_base
        if gcs_prefix:
            trimmed = gcs_prefix.strip().strip("/")
            if trimmed:
                cleaned_prefix = f"{trimmed}/{prefix_base}"
        tasks = start_zone_exports(
            artifacts,
            aoi_name=aoi_name,
            months=used_months,
            bucket=bucket,
            include_stats=include_zonal_stats,
            prefix_override=cleaned_prefix,
        )
        result["bucket"] = bucket
        result["prefix"] = cleaned_prefix
        gcs_components = {
            ext: f"gs://{bucket}/{cleaned_prefix}.{ext}"
            for ext in ["shp", "dbf", "shx", "prj"]
        }
        result["paths"] = {
            "raster": f"gs://{bucket}/{cleaned_prefix}.tif",
            "vectors": gcs_components["shp"],
            "vector_components": gcs_components,
            "zonal_stats": (
                f"gs://{bucket}/{cleaned_prefix}_zonal_stats.csv"
                if include_zonal_stats
                else None
            ),
        }
        result["tasks"] = {
            "raster": _task_payload(tasks.get("raster")),
            "vectors": _task_payload(tasks.get("vectors")),
            "zonal_stats": _task_payload(tasks.get("stats")),
        }
        return result

    folder = (os.getenv("GEE_DRIVE_FOLDER") or "Sentinel2_Indices").strip() or "Sentinel2_Indices"
    folder = folder.rstrip("/")
    if not folder.endswith("zones"):
        folder = f"{folder}/zones"
    drive_prefix = prefix_base.split("/")[-1]
    tasks = start_zone_exports_drive(
        artifacts,
        folder=folder,
        prefix=drive_prefix,
        include_stats=include_zonal_stats,
    )
    result["folder"] = folder
    result["prefix"] = drive_prefix
    drive_components = {
        ext: f"drive://{folder}/{drive_prefix}.{ext}"
        for ext in ["shp", "dbf", "shx", "prj"]
    }
    result["paths"] = {
        "raster": f"drive://{folder}/{drive_prefix}.tif",
        "vectors": drive_components["shp"],
        "vector_components": drive_components,
        "zonal_stats": (
            f"drive://{folder}/{drive_prefix}_zonal_stats.csv"
            if include_zonal_stats
            else None
        ),
    }
    result["tasks"] = {
        "raster": _task_payload(tasks.get("raster")),
        "vectors": _task_payload(tasks.get("vectors")),
        "zonal_stats": _task_payload(tasks.get("stats")),
    }
    return result

