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
    "min": 0.0,
    "max": 1.0,
    "palette": ["440154", "3b528b", "21918c", "5ec962", "fde725"],
}
ZONES_VIS = {
    "min": 1,
    "max": 5,
    "palette": ["440154", "30678d", "35b779", "fde725", "f4f18f"],
}


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
    smooth_radius_m: float = 0,
    mmu_pixels: int = 0,
    band: str = "NDVI",
    n_classes: int = 5,
) -> Dict[str, Any]:
    if method not in ("quantile", "fixed"):
        raise ValueError("method must be 'quantile' or 'fixed'")
    if method == "fixed" and (not fixed_breaks or len(fixed_breaks) != n_classes - 1):
        raise ValueError("fixed_breaks must provide n_classes - 1 thresholds")

    geometry = to_geometry(aoi)
    band_image = image.select(band).clip(geometry)
    if smooth_radius_m > 0:
        band_image = band_image.focal_median(smooth_radius_m, "circle", "meters")

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
        thresholds = [ee.Number(stats.get(name)) for name in output_names]
    else:
        thresholds = [ee.Number(v) for v in fixed_breaks or []]

    if len(thresholds) != n_classes - 1:
        raise ValueError("Unable to compute thresholds for classification.")

    classes = ee.Image.constant(1)
    for idx, threshold in enumerate(thresholds, start=2):
        classes = classes.where(band_image.gte(threshold), idx)

    classes = classes.rename("zone").toInt().clip(geometry)
    if mmu_pixels > 0:
        connected = classes.connectedPixelCount(8, True)
        classes = classes.updateMask(connected.gte(mmu_pixels))

    return {"classes": classes, "thresholds": thresholds, "method": method}


def vectorize_zones(
    classes_img: ee.Image,
    aoi: Mapping[str, Any],
    *,
    simplify_tolerance_m: float = 0,
    eight_connected: bool = True,
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
        if simplify_tolerance_m > 0:
            geom = geom.simplify(simplify_tolerance_m)
        area_ha = geom.area(maxError=1).divide(10_000)
        return (
            ee.Feature(geom)
            .set("zone", zone)
            .set("areaHa", area_ha)
            .copyProperties(feature, exclude=["system:index"])
        )

    return ee.FeatureCollection(vectors.map(_post_process))


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


def _band_stat_keys(band: str) -> Dict[str, str]:
    return {
        "mean": f"{band}_mean",
        "median": f"{band}_median",
        "min": f"{band}_min",
        "max": f"{band}_max",
        "stdDev": f"{band}_stdDev",
        "p10": f"{band}_percentile_10",
        "p25": f"{band}_percentile_25",
        "p75": f"{band}_percentile_75",
        "p90": f"{band}_percentile_90",
    }


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
        for prop, key in mapping.items():
            properties[prop] = stats.get(key)
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
        for prop, key in mapping.items():
            properties[prop] = stats.get(key)
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
    "NDVI_VIS",
    "ZONES_VIS",
    "_ndvi",
    "_osavi",
    "_ndre",
    "_sentinel2_collection",
]
