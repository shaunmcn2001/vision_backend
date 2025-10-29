from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Optional, Sequence, Tuple

import ee

from .earth_engine import ensure_ee
from .zones_workflow import classify_zones, vectorize_zones

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
DEFAULT_SCALE = 10
_CLOUD_CLASSES = (3, 8, 9, 10, 11)
_STAGE_WEIGHTS = {"early": 0.25, "production": 0.5, "late": 0.25}


@dataclass(frozen=True)
class SeasonDefinition:
    sowing_date: date
    harvest_date: date


@dataclass(frozen=True)
class AdvancedZonesResult:
    composite: ee.Image
    zones_raster: ee.Image
    raw_zones: ee.FeatureCollection
    dissolved_zones: ee.FeatureCollection


def _mask_clouds(image: ee.Image) -> ee.Image:
    scl = image.select("SCL")
    mask = ee.Image(1)
    for cls in _CLOUD_CLASSES:
        mask = mask.And(scl.neq(cls))
    return image.updateMask(mask)


def _collection(aoi: ee.Geometry, start: date, end: date) -> ee.ImageCollection:
    ensure_ee()
    return (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(aoi)
        .filterDate(start.isoformat(), (end + timedelta(days=1)).isoformat())
        .map(_mask_clouds)
    )


def _index_image(image: ee.Image, index: str) -> ee.Image:
    if index == "OSAVI":
        expr = "(nir - red) / (nir + red + 0.16)"
        return image.expression(expr, {"nir": image.select("B8"), "red": image.select("B4")}).rename("index")
    if index == "NDVI":
        return image.normalizedDifference(["B8", "B4"]).rename("index")
    if index == "NDRE":
        return image.normalizedDifference(["B8", "B5"]).rename("index")
    raise ValueError(f"Unsupported index: {index}")


def _mean_index(aoi: ee.Geometry, start: date, end: date, index: str) -> Optional[ee.Image]:
    if end <= start:
        end = start + timedelta(days=1)
    collection = _collection(aoi, start, end)
    if collection.size().getInfo() == 0:
        return None
    return collection.map(lambda img: _index_image(img, index)).mean().clip(aoi).rename("index")


def _robust_zscore(image: ee.Image, aoi: ee.Geometry) -> Optional[ee.Image]:
    reducer = ee.Reducer.median().combine(ee.Reducer.percentile([25, 75]), sharedInputs=True)
    stats = image.reduceRegion(
        reducer=reducer,
        geometry=aoi,
        scale=DEFAULT_SCALE,
        bestEffort=True,
        maxPixels=1e13,
        tileScale=4,
    )
    median = stats.get("index_median")
    p25 = stats.get("index_p25")
    p75 = stats.get("index_p75")
    if any(v is None for v in (median, p25, p75)):
        return None
    median_val = ee.Number(median)
    sigma = ee.Number(p75).subtract(ee.Number(p25)).divide(1.349).max(1e-6)
    return image.subtract(median_val).divide(sigma)


def _season_stages(season: SeasonDefinition) -> List[Tuple[str, date, date, str]]:
    sow = season.sowing_date
    harvest = season.harvest_date
    early_end = sow + timedelta(days=40)
    late_start = harvest - timedelta(days=30)
    return [
        ("early", sow, early_end, "OSAVI"),
        ("production", max(early_end, sow), min(late_start, harvest), "NDVI"),
        ("late", late_start, harvest, "NDRE"),
    ]


def _season_score(aoi: ee.Geometry, season: SeasonDefinition) -> Optional[ee.Image]:
    layers: List[Tuple[ee.Image, float]] = []
    for stage, start, end, index in _season_stages(season):
        image = _mean_index(aoi, start, end, index)
        if image is None:
            continue
        blurred = image.focal_gaussian(20, units="meters")
        zscore = _robust_zscore(blurred, aoi)
        if zscore is None:
            continue
        layers.append((zscore.rename("score"), _STAGE_WEIGHTS[stage]))
    if not layers:
        return None
    total_weight = sum(weight for _, weight in layers)
    weighted = layers[0][0].multiply(layers[0][1])
    for image, weight in layers[1:]:
        weighted = weighted.add(image.multiply(weight))
    return weighted.divide(total_weight).rename("score").clip(aoi)


def compute_advanced_layers(
    aoi: ee.Geometry,
    seasons: Sequence[SeasonDefinition],
    breaks: Sequence[float],
) -> AdvancedZonesResult:
    if len(breaks) != 4:
        raise ValueError("Advanced zones require exactly four break thresholds")
    season_scores = [score for season in seasons if (score := _season_score(aoi, season)) is not None]
    if not season_scores:
        raise ValueError("No advanced zone layers could be generated for the supplied seasons.")
    long_term = ee.ImageCollection(season_scores).median().rename("score")
    zones_raster = classify_zones(long_term, aoi, method="fixed", fixed_breaks=list(breaks), band="score")
    raw_vectors = vectorize_zones(zones_raster, aoi, connectivity=8)
    classes = ee.List(raw_vectors.aggregate_array("zone").distinct())

    def _dissolve(zone_value):
        subset = raw_vectors.filter(ee.Filter.eq("zone", zone_value))
        geom = subset.geometry()
        area = geom.area(1.0).divide(10_000)
        return ee.Feature(geom, {"zone": zone_value, "areaHa": area})

    dissolved = ee.FeatureCollection(classes.map(_dissolve))
    return AdvancedZonesResult(
        composite=long_term,
        zones_raster=zones_raster,
        raw_zones=raw_vectors,
        dissolved_zones=dissolved,
    )
