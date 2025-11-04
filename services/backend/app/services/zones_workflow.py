"""Core NDVI and zoning workflows built on Earth Engine."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List, Mapping, Sequence

import ee

from app.services.earth_engine import ensure_ee, to_geometry

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
CLOUD_CLASSES = [3, 8, 9, 10, 11]
DEFAULT_SCALE = 10
NDVI_VIS = {
    "min": -0.2,
    "max": 0.8,
    "palette": [
        "f9f5d7",
        "f6cf75",
        "ee964b",
        "4f9d69",
        "226f54",
        "193b48",
    ],
}
ZONE_BASE_PALETTE = [
    "f5e6b3",
    "d1ce7a",
    "7fb285",
    "4f8f8c",
    "3a6b82",
    "325272",
    "273b5b",
    "1e2a44",
    "151c31",
]
ZONES_VIS = {
    "min": 1,
    "max": 9,
    "palette": ZONE_BASE_PALETTE,
}


def zone_palette(n_classes: int) -> list[str]:
    """Return a palette sized for the requested number of classes."""
    n_classes = max(1, min(n_classes, len(ZONE_BASE_PALETTE)))
    if n_classes <= len(ZONE_BASE_PALETTE):
        return ZONE_BASE_PALETTE[:n_classes]
    return ZONE_BASE_PALETTE


def _mask_sentinel2(image: ee.Image) -> ee.Image:
    scl = image.select("SCL")
    mask = ee.Image(1)
    for cls in CLOUD_CLASSES:
        mask = mask.And(scl.neq(cls))
    return image.updateMask(mask).copyProperties(image, ["system:time_start"])


def _sentinel2_collection(
    geometry: ee.Geometry, start: date, end: date
) -> ee.ImageCollection:
    ensure_ee()
    return (
        ee.ImageCollection(S2_COLLECTION)
        .filterDate(start.isoformat(), end.isoformat())
        .filterBounds(geometry)
        .map(_mask_sentinel2)
    )


def _ndvi(image: ee.Image) -> ee.Image:
    red = image.select("B4").multiply(0.0001)
    nir = image.select("B8").multiply(0.0001)
    ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
    return ndvi.copyProperties(image, ["system:time_start"])


def _osavi(image: ee.Image) -> ee.Image:
    red = image.select("B4").multiply(0.0001)
    nir = image.select("B8").multiply(0.0001)
    osavi = nir.subtract(red).divide(nir.add(red).add(0.16)).rename("OSAVI")
    return osavi.copyProperties(image, ["system:time_start"])


def _ndre(image: ee.Image) -> ee.Image:
    red_edge = image.select("B5").multiply(0.0001)
    nir = image.select("B8").multiply(0.0001)
    ndre = nir.subtract(red_edge).divide(nir.add(red_edge)).rename("NDRE")
    return ndre.copyProperties(image, ["system:time_start"])


def _month_windows(start: date, end: date) -> List[tuple[date, date]]:
    windows: List[tuple[date, date]] = []
    cursor = start.replace(day=1)
    last = end.replace(day=1)
    while cursor <= last:
        next_month_year = cursor.year + (1 if cursor.month == 12 else 0)
        next_month_month = 1 if cursor.month == 12 else cursor.month + 1
        next_month = date(next_month_year, next_month_month, 1)
        window_start = cursor if cursor >= start else start
        window_end = min(next_month, end + timedelta(days=1))
        windows.append((window_start, window_end))
        cursor = next_month
    return windows


def _empty_image(geometry: ee.Geometry, band_name: str) -> ee.Image:
    return ee.Image.constant(0).rename(band_name).clip(geometry)


def monthly_ndvi(aoi: Mapping[str, Any], start: date, end: date) -> ee.ImageCollection:
    geometry = to_geometry(aoi)
    windows = _month_windows(start, end)
    images: List[ee.Image] = []

    for window_start, window_end in windows:
        collection = _sentinel2_collection(geometry, window_start, window_end).map(_ndvi)
        month_image = ee.Image(
            ee.Algorithms.If(
                collection.size().eq(0),
                _empty_image(geometry, "NDVI"),
                collection.mean().rename("NDVI").clip(geometry),
            )
        )
        first_day = window_start
        month_image = ee.Image(month_image).set(
            {
                "year": first_day.year,
                "month": first_day.month,
                "label": f"ndvi_{first_day.strftime('%Y-%m')}",
            }
        )
        images.append(ee.Image(month_image))

    return ee.ImageCollection(images)


def mean_ndvi(aoi: Mapping[str, Any], start: date, end: date) -> ee.Image:
    geometry = to_geometry(aoi)
    exclusive_end = end + timedelta(days=1)
    collection = _sentinel2_collection(geometry, start, exclusive_end).map(_ndvi)
    return ee.Image(
        ee.Algorithms.If(
            collection.size().eq(0),
            _empty_image(geometry, "NDVI"),
            collection.mean().rename("NDVI").clip(geometry),
        )
    )


def classify_zones(
    image: ee.Image,
    aoi: Mapping[str, Any],
    *,
    method: str = "quantile",
    fixed_breaks: Sequence[float] | None = None,
    gaussian_radius_m: float = 0,
    mode_radius_m: float = 0,
    opening_radius_m: float = 0,
    closing_radius_m: float = 0,
    mmu_hectares: float = 0,
    band: str = "NDVI",
    n_classes: int = 5,
) -> Dict[str, Any]:
    if method not in ("quantile", "fixed"):
        raise ValueError("method must be 'quantile' or 'fixed'")
    if method == "fixed" and (not fixed_breaks or len(fixed_breaks) != n_classes - 1):
        raise ValueError("fixed_breaks must provide n_classes - 1 thresholds")

    geometry = to_geometry(aoi)
    band_image = image.select(band).clip(geometry)
    if gaussian_radius_m > 0:
        sigma = max(gaussian_radius_m / 3, 1)
        kernel = ee.Kernel.gaussian(radius=gaussian_radius_m, sigma=sigma, units="meters")
        band_image = band_image.convolve(kernel)

    scale = DEFAULT_SCALE
    if method == "quantile":
        percentiles = [int(i * 100 / n_classes) for i in range(1, n_classes)]
        output_names = [f"p{p}" for p in percentiles]
        reducer = ee.Reducer.percentile(percentiles, outputNames=output_names)
        stats = band_image.reduceRegion(
            reducer=reducer,
            geometry=geometry,
            scale=scale,
            bestEffort=True,
            maxPixels=1e13,
        )
        try:
            stats_info = stats.getInfo()
        except ee.EEException as exc:
            raise ValueError("Unable to compute thresholds for classification.") from exc
        if not isinstance(stats_info, dict) or not stats_info:
            raise ValueError("Insufficient data to compute zone percentiles; try a different date range or AOI.")
        band_names = []
        try:
            band_names = band_image.bandNames().getInfo() or []
        except ee.EEException:
            band_names = []
        values: list[float] = []
        for name in output_names:
            value = stats_info.get(name)
            if value is None:
                for band in band_names:
                    value = stats_info.get(f"{band}_{name}")
                    if value is not None:
                        break
            if value is None:
                raise ValueError("Insufficient data to compute zone percentiles; try a different date range or AOI.")
            values.append(float(value))
        thresholds = [ee.Number(v) for v in values]
    else:
        thresholds = [ee.Number(v) for v in fixed_breaks or []]

    if len(thresholds) != n_classes - 1:
        raise ValueError("Unable to compute thresholds for classification.")

    classes = ee.Image.constant(1)
    for idx, threshold in enumerate(thresholds, start=2):
        classes = classes.where(band_image.gte(threshold), idx)

    classes = classes.rename("zone").toInt().clip(geometry)
    if mode_radius_m > 0:
        classes = (
            classes.focal_mode(mode_radius_m, "circle", "meters")
            .reproject(crs=classes.projection(), scale=DEFAULT_SCALE)
            .toInt()
        )
    if opening_radius_m > 0:
        classes = (
            classes.focal_min(opening_radius_m, "circle", "meters")
            .focal_max(opening_radius_m, "circle", "meters")
            .reproject(crs=classes.projection(), scale=DEFAULT_SCALE)
            .toInt()
        )
    if closing_radius_m > 0:
        classes = (
            classes.focal_max(closing_radius_m, "circle", "meters")
            .focal_min(closing_radius_m, "circle", "meters")
            .reproject(crs=classes.projection(), scale=DEFAULT_SCALE)
            .toInt()
        )

    if mmu_hectares > 0:
        min_pixels = ee.Number(mmu_hectares).multiply(10000.0 / (DEFAULT_SCALE ** 2))
        connected = (
            classes.connectedPixelCount(8, True)
            .reproject(crs=classes.projection(), scale=DEFAULT_SCALE)
        )
        mask = connected.gte(min_pixels)
        fallback_radius = mode_radius_m or 30
        fallback = (
            classes.focal_mode(fallback_radius, "circle", "meters")
            .reproject(crs=classes.projection(), scale=DEFAULT_SCALE)
            .toInt()
        )
        classes = classes.updateMask(mask).unmask(fallback).toInt()

    return {"classes": classes.clip(geometry).toInt(), "thresholds": thresholds, "method": method}


def vectorize_zones(
    classes_img: ee.Image,
    aoi: Mapping[str, Any],
    *,
    simplify_tolerance_m: float = 0,
    eight_connected: bool = True,
    smooth_buffer_m: float = 0,
    min_area_hectares: float = 0,
) -> ee.FeatureCollection:
    geometry = to_geometry(aoi)
    label_image = classes_img.rename("zone").toInt().clip(geometry)
    vectors = label_image.reduceToVectors(
        geometry=geometry,
        scale=DEFAULT_SCALE,
        maxPixels=1e13,
        geometryType="polygon",
        labelProperty="zone",
        eightConnected=eight_connected,
    )

    def _post_process(feature: ee.Feature) -> ee.Feature:
        zone = ee.Number(feature.get("zone")).toInt()
        geom = feature.geometry()
        if smooth_buffer_m > 0:
            geom = geom.buffer(smooth_buffer_m).buffer(-smooth_buffer_m)
        if simplify_tolerance_m > 0:
            geom = geom.simplify(simplify_tolerance_m)
        area_ha = geom.area(maxError=1).divide(10_000)
        return (
            ee.Feature(geom)
            .set("zone", zone)
            .set("areaHa", area_ha)
            .copyProperties(feature, exclude=["system:index"])
        )

    collection = ee.FeatureCollection(vectors.map(_post_process))
    if min_area_hectares > 0:
        collection = collection.filter(ee.Filter.gte("areaHa", min_area_hectares))
    return collection


def dissolve_by_class(
    feature_collection: ee.FeatureCollection, property_name: str = "zone"
) -> ee.FeatureCollection:
    classes = ee.Dictionary(
        feature_collection.aggregate_histogram(property_name)
    ).keys()

    def _dissolve(value: Any) -> ee.Feature:
        zone_value = ee.Number.parse(value)
        geom = (
            feature_collection.filter(
                ee.Filter.eq(property_name, zone_value)
            ).geometry(maxError=1)
        ).dissolve()
        area_ha = geom.area(maxError=1).divide(10_000)
        return ee.Feature(geom).set(property_name, zone_value).set("areaHa", area_ha)

    return ee.FeatureCollection(ee.List(classes).map(_dissolve))


def _band_stat_keys(band: str) -> Dict[str, Sequence[str]]:
    return {
        "mean": [f"{band}_mean"],
        "median": [f"{band}_median"],
        "min": [f"{band}_min"],
        "max": [f"{band}_max"],
        "stdDev": [f"{band}_stdDev"],
        "p10": [f"{band}_percentile_10", f"{band}_p10"],
        "p25": [f"{band}_percentile_25", f"{band}_p25"],
        "p75": [f"{band}_percentile_75", f"{band}_p75"],
        "p90": [f"{band}_percentile_90", f"{band}_p90"],
    }


def _first_present_stat(stats: ee.Dictionary, keys: Sequence[str]) -> ee.Number:
    value: ee.ComputedObject = ee.Number(0)
    for key in keys:
        value = ee.Number(ee.Algorithms.If(stats.contains(key), stats.get(key), value))
    return ee.Number(value)


def zone_statistics(
    image: ee.Image,
    classes_img: ee.Image,
    aoi: Mapping[str, Any],
    *,
    band: str = "NDVI",
    class_values: Sequence[int] = (1, 2, 3, 4, 5),
) -> ee.FeatureCollection:
    geometry = to_geometry(aoi)
    zone_band = classes_img.rename("zone").toInt()

    reducer = (
        ee.Reducer.mean()
        .combine(ee.Reducer.median(), sharedInputs=True)
        .combine(ee.Reducer.min(), sharedInputs=True)
        .combine(ee.Reducer.max(), sharedInputs=True)
        .combine(ee.Reducer.stdDev(), sharedInputs=True)
        .combine(ee.Reducer.percentile([10, 25, 75, 90]), sharedInputs=True)
    )
    mapping = _band_stat_keys(band)

    def _zone_feature(value: Any) -> ee.Feature:
        zone_value = ee.Number(value)
        mask = zone_band.eq(zone_value).rename("mask")
        masked = image.select(band).updateMask(mask)

        stats = masked.reduceRegion(
            reducer=reducer,
            geometry=geometry,
            scale=DEFAULT_SCALE,
            bestEffort=True,
            maxPixels=1e13,
        )
        pixel_count = mask.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geometry,
            scale=DEFAULT_SCALE,
            bestEffort=True,
            maxPixels=1e13,
        ).get("mask")
        area = (
            ee.Image.pixelArea()
            .updateMask(mask)
            .reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=geometry,
                scale=DEFAULT_SCALE,
                bestEffort=True,
                maxPixels=1e13,
            )
            .get("area")
        )

        properties = {
            "zone": zone_value,
            "pixelCount": ee.Number(pixel_count).toInt(),
            "areaHa": ee.Number(area).divide(10_000),
        }
        stats_dict = ee.Dictionary(stats)
        for prop, keys in mapping.items():
            properties[prop] = _first_present_stat(stats_dict, keys)
        return ee.Feature(None, properties)

    features = ee.FeatureCollection(
        ee.List(list(class_values)).map(_zone_feature)
    ).filter(ee.Filter.gt("pixelCount", 0))
    return features


def dissolved_zone_statistics(
    image: ee.Image,
    classes_img: ee.Image,
    dissolved_fc: ee.FeatureCollection,
    *,
    band: str = "NDVI",
    class_property: str = "zone",
) -> ee.FeatureCollection:
    mapping = _band_stat_keys(band)

    def _stats(feature: ee.Feature) -> ee.Feature:
        zone_value = ee.Number(feature.get(class_property))
        geom = feature.geometry()
        mask = classes_img.eq(zone_value).rename("mask")
        masked = image.select(band).updateMask(mask)
        stats = masked.reduceRegion(
            reducer=(
                ee.Reducer.mean()
                .combine(ee.Reducer.median(), sharedInputs=True)
                .combine(ee.Reducer.min(), sharedInputs=True)
                .combine(ee.Reducer.max(), sharedInputs=True)
                .combine(ee.Reducer.stdDev(), sharedInputs=True)
                .combine(ee.Reducer.percentile([10, 25, 75, 90]), sharedInputs=True)
            ),
            geometry=geom,
            scale=DEFAULT_SCALE,
            bestEffort=True,
            maxPixels=1e13,
        )
        pixel_count = mask.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geom,
            scale=DEFAULT_SCALE,
            bestEffort=True,
            maxPixels=1e13,
        ).get("mask")
        area_ha = geom.area(maxError=1).divide(10_000)
        properties = {
            class_property: zone_value,
            "pixelCount": ee.Number(pixel_count).toInt(),
            "areaHa": area_ha,
        }
        stats_dict = ee.Dictionary(stats)
        for prop, keys in mapping.items():
            properties[prop] = _first_present_stat(stats_dict, keys)
        return feature.set(properties)

    return ee.FeatureCollection(dissolved_fc.map(_stats)).filter(
        ee.Filter.gt("pixelCount", 0)
    )


__all__ = [
    "monthly_ndvi",
    "mean_ndvi",
    "classify_zones",
    "vectorize_zones",
    "dissolve_by_class",
    "zone_statistics",
    "dissolved_zone_statistics",
    "zone_palette",
    "NDVI_VIS",
    "ZONES_VIS",
    "_ndvi",
    "_osavi",
    "_ndre",
    "_sentinel2_collection",
]
