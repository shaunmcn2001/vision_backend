"""Production zone workflow built on Sentinel-2 monthly composites."""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, Iterable, List, Sequence

import ee

from app import gee, indices
from app.exports import sanitize_name


DEFAULT_ZONE_INDICES = ["NDVI", "NDRE", "NDMI", "BSI"]
DEFAULT_CLOUD_PROB_MAX = 40
DEFAULT_K_ZONES = 3
DEFAULT_CV_THRESHOLD = 0.25
DEFAULT_MIN_MAPPING_UNIT_HA = 0.5
DEFAULT_SAMPLE_SIZE = 8000
DEFAULT_SCALE = 10
DEFAULT_CRS = "EPSG:4326"


@dataclass(frozen=True)
class ZoneArtifacts:
    """Container for the images/vectors used for production zone exports."""

    zone_image: ee.Image
    zone_vectors: ee.FeatureCollection
    zonal_stats: ee.FeatureCollection | None
    mean_images: Dict[str, ee.Image]
    geometry: ee.Geometry


def _ensure_indices(indices_for_zoning: Sequence[str]) -> List[str]:
    """Return a unique, case-sensitive list of indices ensuring NDVI is present."""

    canonical_lookup = {name.lower(): name for name in indices.SUPPORTED_INDICES}
    resolved: List[str] = []
    for name in indices_for_zoning:
        key = str(name).strip()
        if not key:
            continue
        canonical = canonical_lookup.get(key.lower())
        if canonical is None:
            raise ValueError(f"Unsupported index for zoning: {name}")
        if canonical not in resolved:
            resolved.append(canonical)

    if "NDVI" not in resolved:
        resolved.insert(0, "NDVI")

    return resolved


def _month_bounds(months: Sequence[str]) -> tuple[str, str]:
    if not months:
        raise ValueError("At least one month must be supplied")
    ordered = list(dict.fromkeys(months))
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


def _normalise_feature(mean_image: ee.Image, geometry: ee.Geometry, name: str) -> ee.Image:
    band_name = ee.String(mean_image.bandNames().get(0))
    stats = mean_image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=DEFAULT_SCALE,
        bestEffort=True,
        tileScale=4,
    )
    std_stats = mean_image.reduceRegion(
        reducer=ee.Reducer.stdDev(),
        geometry=geometry,
        scale=DEFAULT_SCALE,
        bestEffort=True,
        tileScale=4,
    )
    mean_value = ee.Number(stats.get(band_name, 0))
    std_value = ee.Number(std_stats.get(band_name, 0)).max(1e-6)
    return mean_image.subtract(mean_value).divide(ee.Image.constant(std_value)).rename(f"norm_{name}")


def _stability_mask(cv_images: Iterable[ee.Image], threshold: float) -> ee.Image:
    mask = ee.Image.constant(1)
    for cv_image in cv_images:
        mask = mask.And(cv_image.lte(threshold))
    return mask


def _rank_zones(cluster_image: ee.Image, ndvi_mean: ee.Image, geometry: ee.Geometry) -> ee.Image:
    cluster_band = cluster_image.rename("cluster")
    stats_image = cluster_band.addBands(ndvi_mean.rename("mean_ndvi"))
    grouped = stats_image.reduceRegion(
        reducer=ee.Reducer.mean().group(groupField=0, groupName="cluster"),
        geometry=geometry,
        scale=DEFAULT_SCALE,
        bestEffort=True,
        tileScale=4,
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


def _add_zonal_stats(
    feature: ee.Feature,
    mean_images: Dict[str, ee.Image],
) -> ee.Feature:
    geometry = feature.geometry()
    area_ha = geometry.area(maxError=1).divide(10_000)
    stats_image = ee.Image.cat([mean_images[name] for name in sorted(mean_images)])
    stats = stats_image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=DEFAULT_SCALE,
        bestEffort=True,
        tileScale=4,
    )
    return feature.set(stats).set({"area_ha": area_ha})


def build_zone_artifacts(
    aoi_geojson: dict,
    *,
    months: Sequence[str],
    indices_for_zoning: Sequence[str] = DEFAULT_ZONE_INDICES,
    cloud_prob_max: int = DEFAULT_CLOUD_PROB_MAX,
    k_zones: int = DEFAULT_K_ZONES,
    cv_mask_threshold: float = DEFAULT_CV_THRESHOLD,
    min_mapping_unit_ha: float = DEFAULT_MIN_MAPPING_UNIT_HA,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    include_stats: bool = True,
) -> ZoneArtifacts:
    if k_zones < 2:
        raise ValueError("k_zones must be at least 2")
    if min_mapping_unit_ha <= 0:
        raise ValueError("min_mapping_unit_ha must be positive")

    gee.initialize()
    geometry = gee.geometry_from_geojson(aoi_geojson)

    resolved_indices = _ensure_indices(indices_for_zoning)

    monthly_composites: List[ee.Image] = []
    for month in dict.fromkeys(months):
        _, composite = gee.monthly_sentinel2_collection(geometry, month, cloud_prob_max)
        monthly_composites.append(composite)

    if not monthly_composites:
        raise ValueError("No monthly composites were generated")

    index_collections: Dict[str, List[ee.Image]] = {name: [] for name in resolved_indices}
    for composite in monthly_composites:
        for name in resolved_indices:
            image = indices.compute_index(composite, name, geometry, DEFAULT_SCALE)
            index_collections[name].append(image)

    mean_images: Dict[str, ee.Image] = {}
    cv_images: Dict[str, ee.Image] = {}

    for name, images in index_collections.items():
        if not images:
            raise ValueError(f"No imagery available for index {name}")
        collection = ee.ImageCollection(images)
        mean_image = collection.mean().rename(f"mean_{name}")
        std_image = collection.reduce(ee.Reducer.stdDev()).rename(f"std_{name}")
        mean_abs = mean_image.abs()
        cv_image = std_image.divide(mean_abs.where(mean_abs.gt(1e-6), ee.Image.constant(1e-6)))
        cv_image = cv_image.updateMask(mean_abs.gt(1e-6)).rename(f"cv_{name}")
        mean_images[name] = mean_image
        cv_images[name] = cv_image

    stability = _stability_mask(cv_images.values(), cv_mask_threshold)

    normalised_bands: List[ee.Image] = []
    for name, mean_image in mean_images.items():
        normalized = _normalise_feature(mean_image, geometry, name)
        normalised_bands.append(normalized)

    feature_stack = ee.Image.cat(normalised_bands).updateMask(stability)

    training = feature_stack.sample(
        region=geometry,
        scale=DEFAULT_SCALE,
        numPixels=sample_size,
        seed=42,
        tileScale=4,
        geometries=False,
    )
    clusterer = ee.Clusterer.wekaKMeans(k_zones).train(training)
    clustered = feature_stack.cluster(clusterer)

    if "NDVI" not in mean_images:
        raise ValueError("NDVI mean image is required for ranking")

    ranked = _rank_zones(clustered, mean_images["NDVI"], geometry).updateMask(stability)

    smoothed = ranked.focal_mode(radius=1, units="pixels", iterations=1).updateMask(ranked.mask())
    pixel_area = ee.Image.pixelArea()
    component_area = (
        smoothed.connectedPixelCount(maxSize=1_000_000, eightConnected=True).multiply(pixel_area)
    )
    min_area_m2 = min_mapping_unit_ha * 10_000
    cleaned = smoothed.updateMask(component_area.gte(min_area_m2)).rename("zone")
    zone_image = cleaned.toInt16()

    vectors = zone_image.reduceToVectors(
        geometry=geometry,
        scale=DEFAULT_SCALE,
        maxPixels=gee.MAX_PIXELS,
        geometryType="polygon",
        eightConnected=True,
        labelProperty="zone",
        reducer=ee.Reducer.first(),
    )

    stats_collection = None
    if include_stats:
        def _mapper(feature):
            result = _add_zonal_stats(feature, mean_images)
            return result.set({"zone": ee.Number(feature.get("zone")).toInt()})

        stats_collection = ee.FeatureCollection(vectors.map(_mapper))

    return ZoneArtifacts(
        zone_image=zone_image,
        zone_vectors=vectors,
        zonal_stats=stats_collection,
        mean_images=mean_images,
        geometry=geometry,
    )


def start_zone_exports(
    artifacts: ZoneArtifacts,
    *,
    aoi_name: str,
    months: Sequence[str],
    bucket: str,
    include_stats: bool = True,
) -> Dict[str, ee.batch.Task]:
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

