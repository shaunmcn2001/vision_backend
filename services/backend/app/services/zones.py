"""Production zone workflow built on Sentinel-2 monthly composites."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import calendar
import os
import math
from typing import Dict, List, Mapping, Sequence, Tuple, Union

import ee

from app import gee
from app.exports import sanitize_name
from app.utils.geometry import area_ha


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
    if isinstance(aoi, ee.Geometry):
        return aoi
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
    raw_mean = collection.mean()
    mean = raw_mean.rename("NDVI_mean")
    median = collection.median().rename("NDVI_median")
    raw_std = collection.reduce(ee.Reducer.stdDev())
    std = raw_std.rename("NDVI_stdDev")

    positive_mask = raw_mean.gt(0)
    cv_raw = raw_std.divide(raw_mean)
    cv = (
        cv_raw.where(raw_mean.lte(0), 0)
        .updateMask(positive_mask)
        .rename("NDVI_cv")
    )

    mean = mean.updateMask(positive_mask)
    median = median.updateMask(positive_mask)
    std = std.updateMask(positive_mask)

    return {"mean": mean, "median": median, "std": std, "cv": cv}


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
    except Exception as exc:  # pragma: no cover
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
    stats = stats_image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=DEFAULT_SCALE,
        bestEffort=True,
        tileScale=4,
        maxPixels=gee.MAX_PIXELS,
    )
    zone_value = ee.Number(feature.get("zone")).toInt()
    return feature.set(stats).set({"area_ha": area_ha_val, "zone": zone_value, "zone_id": zone_value})


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
        percentile_thresholds: List[float] = [
            float(value) for value in (thresholds.getInfo() or [])
        ]
    except Exception as exc:  # pragma: no cover
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
    smooth_radius_m: float,
    open_radius_m: float,
    close_radius_m: float,
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
        smooth_radius_m=smooth_radius_m,
        open_radius_m=open_radius_m,
        close_radius_m=close_radius_m,
        min_mapping_unit_ha=min_mapping_unit_ha,
    )

    return cleaned.rename("zone"), mean_images


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
        smooth_radius_m=smooth_radius_m,
        open_radius_m=open_radius_m,
        close_radius_m=close_radius_m,
        min_mapping_unit_ha=min_mapping_unit_ha,
    )

    return cleaned.rename("zone"), masked_features


def _prepare_selected_period_artifacts(
    aoi_geojson: Union[dict, ee.Geometry],
    *,
    geometry: ee.Geometry,
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
    stability_enabled = APPLY_STABILITY if apply_stability_mask is None else bool(apply_stability_mask)
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
    stats = _ndvi_temporal_stats(ndvi_images)
    cv_image = stats["cv"]

    mean_pixel_count = _pixel_count(
        stats["mean"],
        geometry,
        context="NDVI mean pixel count",
        scale=DEFAULT_SCALE,
    )
    print("NDVI mean pixel count:", mean_pixel_count)

    initial_threshold = float(cv_mask_threshold)
    thresholds_to_try: List[float] = [initial_threshold]
    for fallback in STABILITY_THRESHOLD_SEQUENCE:
        if fallback > initial_threshold + 1e-9:
            thresholds_to_try.append(fallback)

    total_pixels = _pixel_count(
        cv_image,
        geometry,
        context="stability baseline pixel count",
        scale=DEFAULT_SCALE,
    )

    stability: ee.Image | None = None
    final_threshold = initial_threshold
    survival_ratio = 0.0
    surviving_pixels = 0
    thresholds_tested: List[float] = []
    ratio_history: List[float] = []

    if total_pixels > 0:
        for idx, threshold in enumerate(thresholds_to_try):
            thresholds_tested.append(float(threshold))
            candidate = cv_image.lte(threshold).selfMask()
            surviving_pixels = _pixel_count(
                candidate,
                geometry,
                context="stability surviving pixel count",
                scale=DEFAULT_SCALE,
            )
            ratio = (surviving_pixels / total_pixels) if total_pixels else 0.0
            ratio_history.append(ratio)
            stability = candidate
            final_threshold = float(threshold)
            survival_ratio = ratio
            if ratio >= MIN_STABILITY_SURVIVAL_RATIO or idx == len(thresholds_to_try) - 1:
                break
    else:
        thresholds_tested.append(float(initial_threshold))
        ratio_history.append(0.0)
        stability = ee.Image(1)
        surviving_pixels = 0
        survival_ratio = 0.0

    thresholds_for_mask = thresholds_tested or [initial_threshold]
    guarded_mask = _stability_mask(
        cv_image,
        geometry,
        thresholds_for_mask,
        MIN_STABILITY_SURVIVAL_RATIO,
        DEFAULT_SCALE,
    )

    if stability_enabled:
        tmp_mask = guarded_mask
    else:
        tmp_mask = ee.Image(1)

    mask_count_image = (
        tmp_mask if stability_enabled else tmp_mask.updateMask(cv_image.mask())
    )
    mask_pixel_count = _pixel_count(
        mask_count_image,
        geometry,
        context="stability mask pixel count",
        scale=DEFAULT_SCALE,
    )
    print("Stability mask pixel count:", mask_pixel_count)

    if stability_enabled:
        stability = guarded_mask
    else:
        stability = tmp_mask
        surviving_pixels = mask_pixel_count
        survival_ratio = 1.0 if total_pixels > 0 else 0.0
        final_threshold = initial_threshold

    low_confidence = final_threshold > initial_threshold + 1e-9

    stats = {**stats, "stability": stability}

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

    method_key = method.strip().lower()
    extra_means: Dict[str, ee.Image] = {}

    percentile_thresholds: List[float] = []

    if method_key == "ndvi_percentiles":
        try:
            zone_image, percentile_thresholds = _build_percentile_zones(
                ndvi_stats=stats,
                geometry=geometry,
                n_classes=n_classes,
                smooth_radius_m=smooth_radius_m,
                open_radius_m=open_radius_m,
                close_radius_m=close_radius_m,
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
            smooth_radius_m=smooth_radius_m,
            open_radius_m=open_radius_m,
            close_radius_m=close_radius_m,
            min_mapping_unit_ha=mmu_value,
            sample_size=sample_size,
        )
        extra_means = {
            name: image.updateMask(stats["stability"])
            for name, image in mean_images.items()
        }

    zone_image = zone_image.updateMask(zone_image.neq(0)).toInt16()
    # Do NOT reproject to EPSG:4326 here; keep native meter-scale projection.
    # (Exports will set CRS explicitly.)
    zone_image = zone_image.rename("zone")

    vectors = _prepare_vectors(
        zone_image,
        geometry,
        tolerance_m=simplify_tol_m,
        buffer_m=simplify_buffer_m,
    )

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

    metadata.update(composite_metadata)

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
        "mean_pixel_count": mean_pixel_count,
        "mask_pixel_count": mask_pixel_count,
        "apply_stability": stability_enabled,
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
        "mean_pixel_count": mean_pixel_count,
        "mask_pixel_count": mask_pixel_count,
        "apply_stability": stability_enabled,
    }

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

    method_key = method.strip().lower()
    if method_key not in {"ndvi_percentiles", "multiindex_kmeans"}:
        raise ValueError("Unsupported method for production zones")

    gee.initialize()
    geometry = _resolve_geometry(aoi_geojson)
    start_date, end_date = _month_range_dates(months)

    artifacts, _metadata = _prepare_selected_period_artifacts(
        aoi_geojson,
        geometry=geometry,
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
        crs=DEFAULT_EXPORT_CRS,  # metric CRS to match 10 m scale
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
        crs=DEFAULT_EXPORT_CRS,  # metric CRS to match 10 m scale
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
    aoi_geojson: Union[dict, ee.Geometry],
    aoi_name: str,
    months: Sequence[str] | None,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    geometry: ee.Geometry | None = None,
    cloud_prob_max: int = DEFAULT_CLOUD_PROB_MAX,
    n_classes: int = DEFAULT_N_CLASSES,
    cv_mask_threshold: float = DEFAULT_CV_THRESHOLD,
    mmu_ha: float = DEFAULT_MIN_MAPPING_UNIT_HA,
    smooth_radius_m: float = DEFAULT_SMOOTH_RADIUS_M,
    open_radius_m: float = DEFAULT_OPEN_RADIUS_M,
    close_radius_m: float = DEFAULT_CLOSE_RADIUS_M,
    simplify_tol_m: float = DEFAULT_SIMPLIFY_TOL_M,
    simplify_buffer_m: float = DEFAULT_SIMPLIFY_BUFFER_M,
    export_target: str = "zip",
    gcs_bucket: str | None = None,
    gcs_prefix: str | None = None,
    include_zonal_stats: bool = True,
    apply_stability_mask: bool | None = None,
) -> Dict[str, object]:
    if start_date is not None and end_date is not None and end_date < start_date:
        raise ValueError("end_date must be on or after start_date")

    if months and not isinstance(months, Sequence):
        months = list(months)

    if not months:
        if start_date is None or end_date is None:
            raise ValueError("Either months or start/end dates must be supplied")
        months = _months_from_dates(start_date, end_date)

    ordered_months = _ordered_months(months)
    if start_date is None or end_date is None:
        start_date, end_date = _month_range_dates(ordered_months)
    if n_classes < 3 or n_classes > 7:
        raise ValueError("n_classes must be between 3 and 7")
    if smooth_radius_m < 0 or open_radius_m < 0 or close_radius_m < 0:
        raise ValueError("Smoothing radii must be non-negative")
    if mmu_ha < 0:
        raise ValueError("mmu_ha must be non-negative")

    gee.initialize()
    geometry = geometry or _resolve_geometry(aoi_geojson)

    artifacts, metadata = _prepare_selected_period_artifacts(
        aoi_geojson,
        geometry=geometry,
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
