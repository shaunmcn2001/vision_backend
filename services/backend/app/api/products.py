"""Product endpoints for Vision backend."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Dict, Iterable, List, Mapping

import ee
from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    AdvancedDownloads,
    AdvancedPreview,
    AdvancedZonesRequest,
    AdvancedZonesResponse,
    BasicDownloads,
    BasicZonesRequest,
    BasicZonesResponse,
    ImageryDailyRequest,
    ImageryDailyResponse,
    ImageryDayItem,
    ImagerySummary,
    NDVIMonthItem,
    NDVIMonthRequest,
    NDVIMonthResponse,
    NDVIMonthlyAverage,
    NDVIYearlyAverage,
    TileResponse,
)
from app.services.advanced_zones import SeasonDefinition, compute_advanced_layers
from app.services.downloads import (
    DownloadTooLargeError,
    image_geotiff_url,
    table_csv_url,
    table_shp_url,
)
from app.services.earth_engine import ensure_ee, to_ee_geometry
from app.services.tiles import create_tile_session
from app.services.zones_workflow import (
    NDVI_VIS,
    ZONES_VIS,
    classify_zones,
    dissolve_by_class,
    dissolved_zone_statistics,
    NdviMonthlyStat,
    csv_data_url,
    compute_yearly_ndvi_averages,
    last_year_monthly_stats,
    mean_ndvi,
    monthly_ndvi,
    monthly_stats_to_rows,
    zone_palette,
    vectorize_zones,
    yearly_averages_to_rows,
    zone_statistics,
)

router = APIRouter(prefix="/api", tags=["products"])

logger = logging.getLogger(__name__)

RGB_BANDS = ["B4", "B3", "B2"]
to_geometry = to_ee_geometry  # Backwards compatibility for older imports
_NOOP_REDUCER: object = object()


def _mean_reducer() -> object:
    try:
        return ee.Reducer.mean()
    except Exception:  # pragma: no cover - EE not initialised in tests
        return _NOOP_REDUCER


def _class_stats_fc(
    mean_image,
    classes_img,
    aoi,
    *,
    class_values,
) -> object:
    return zone_statistics(
        mean_image,
        classes_img,
        aoi,
        class_values=class_values,
    )


def _attach_stats(
    composite_img,
    zones_img,
    aoi,
    *,
    dissolved_vectors,
    band: str,
    class_values,
) -> dict[str, object]:
    raw_stats = zone_statistics(
        composite_img,
        zones_img,
        aoi,
        band=band,
        class_values=class_values,
    )
    dissolved_stats = dissolved_zone_statistics(
        composite_img,
        zones_img,
        dissolved_vectors,
        band=band,
    )
    return {"raw": raw_stats, "dissolved": dissolved_stats}


def _month_sequence(start: date, end: date) -> List[date]:
    months: List[date] = []
    cursor = date(start.year, start.month, 1)
    last = date(end.year, end.month, 1)
    while cursor <= last:
        months.append(cursor)
        next_year = cursor.year + (1 if cursor.month == 12 else 0)
        next_month = 1 if cursor.month == 12 else cursor.month + 1
        cursor = date(next_year, next_month, 1)
    return months


def _tile_response(tile: Dict[str, object]) -> TileResponse:
    return TileResponse.parse_obj(tile)


def _mean_ndvi_value(image: ee.Image, geometry: ee.Geometry) -> float | None:
    reducer = _mean_reducer()
    if reducer is _NOOP_REDUCER:
        return None
    try:
        stats = image.reduceRegion(
            reducer=reducer,
            geometry=geometry,
            scale=10,
            bestEffort=True,
            maxPixels=1e13,
        )
        stats_info = stats.getInfo() if hasattr(stats, "getInfo") else stats
    except Exception:  # pragma: no cover - EE errors bubble up in production
        return None
    if not isinstance(stats_info, Mapping):
        return None
    value = stats_info.get("NDVI") or stats_info.get("ndvi")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@router.post("/ndvi/month", response_model=NDVIMonthResponse)
@router.post("/products/ndvi-month", response_model=NDVIMonthResponse)
def ndvi_month(request: NDVIMonthRequest) -> NDVIMonthResponse:
    if request.end < request.start:
        raise HTTPException(status_code=400, detail="end must be on or after start")

    geometry = to_geometry(request.aoi)
    months = _month_sequence(request.start, request.end)
    collection = monthly_ndvi(request.aoi, request.start, request.end)
    collection_size = int(collection.size().getInfo())
    image_list = collection.toList(collection_size)
    vis_params = dict(NDVI_VIS)
    if request.clamp:
        vis_params["min"] = float(request.clamp[0])
        vis_params["max"] = float(request.clamp[1])

    items: List[NDVIMonthItem] = []
    downloads: Dict[str, str] = {}
    monthly_stats: List[NdviMonthlyStat] = []
    for idx, month_start in enumerate(months):
        if idx >= collection_size:
            break
        image = ee.Image(image_list.get(idx))
        label = f"ndvi_{month_start.strftime('%Y-%m')}"
        tile_info = create_tile_session(image, vis_params=vis_params)
        file_stub = label.replace("-", "_")
        try:
            downloads[f"{label}_raw_geotiff"] = image_geotiff_url(
                image,
                request.aoi,
                name=f"{file_stub}_raw_ndvi",
                scale=10,
            )
            colour_image = image.visualize(**vis_params)
            downloads[f"{label}_colour_geotiff"] = image_geotiff_url(
                colour_image,
                request.aoi,
                name=f"{file_stub}_colour_ndvi",
                scale=10,
            )
        except DownloadTooLargeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        mean_value = _mean_ndvi_value(image, geometry)
        items.append(
            NDVIMonthItem(
                name=label,
                tile=_tile_response(tile_info),
                mean_ndvi=mean_value,
            )
        )
        monthly_stats.append(
            NdviMonthlyStat(
                year=month_start.year,
                month=month_start.month,
                label=label,
                mean_ndvi=mean_value,
            )
        )

    mean_image = mean_ndvi(request.aoi, request.start, request.end)
    mean_tile = create_tile_session(mean_image, vis_params=vis_params)
    overall_mean_value = _mean_ndvi_value(mean_image, geometry)
    try:
        downloads["ndvi_mean_raw_geotiff"] = image_geotiff_url(
            mean_image,
            request.aoi,
            name="ndvi_mean_raw",
            scale=10,
        )
        colour_mean = mean_image.visualize(**vis_params)
        downloads["ndvi_mean_colour_geotiff"] = image_geotiff_url(
            colour_mean,
            request.aoi,
            name="ndvi_mean_colour",
            scale=10,
        )
    except DownloadTooLargeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    yearly_averages = compute_yearly_ndvi_averages(monthly_stats)
    last_year_monthly = last_year_monthly_stats(monthly_stats)

    csv_downloads: Dict[str, str] = {}
    monthly_rows = monthly_stats_to_rows(monthly_stats)
    if monthly_rows:
        csv_downloads["ndvi_monthly_stats_csv"] = csv_data_url(
            ["year", "month", "label", "mean_ndvi"], monthly_rows
        )
    yearly_rows = yearly_averages_to_rows(yearly_averages)
    if yearly_rows:
        csv_downloads["ndvi_yearly_stats_csv"] = csv_data_url(
            ["year", "mean_ndvi"], yearly_rows
        )
    last_year_rows = monthly_stats_to_rows(last_year_monthly)
    if last_year_rows:
        csv_downloads["ndvi_last_year_monthly_stats_csv"] = csv_data_url(
            ["year", "month", "label", "mean_ndvi"], last_year_rows
        )

    return NDVIMonthResponse(
        items=items,
        mean=_tile_response(mean_tile),
        downloads=downloads,
        overall_mean_ndvi=overall_mean_value,
        yearly_averages=[
            NDVIYearlyAverage(year=avg.year, mean_ndvi=avg.mean_ndvi)
            for avg in yearly_averages
        ],
        last_year_monthly_averages=[
            NDVIMonthlyAverage(
                year=stat.year,
                month=stat.month,
                label=stat.label,
                mean_ndvi=stat.mean_ndvi,
            )
            for stat in last_year_monthly
        ],
        csv_downloads=csv_downloads,
    )


def _iterate_days(start: date, end: date) -> Iterable[date]:
    cursor = start
    while cursor <= end:
        yield cursor
        cursor += timedelta(days=1)


@router.post("/imagery/daily", response_model=ImageryDailyResponse)
@router.post("/products/imagery/daily", response_model=ImageryDailyResponse)
def imagery_daily(request: ImageryDailyRequest) -> ImageryDailyResponse:
    if request.end < request.start:
        raise HTTPException(status_code=400, detail="end must be on or after start")

    ensure_ee()
    bands = request.bands or RGB_BANDS
    geometry = to_geometry(request.aoi)
    days: List[ImageryDayItem] = []
    cloud_values: List[float] = []

    for day in _iterate_days(request.start, request.end):
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate(day.isoformat(), (day + timedelta(days=1)).isoformat())
            .filterBounds(geometry)
        )
        size = collection.size().getInfo()
        tile = None
        cloud_pct_value = 0.0
        if size:
            mosaic = collection.mosaic().clip(geometry)
            tile_info = create_tile_session(
                mosaic.select(bands),
                vis_params={"bands": bands, "min": 0, "max": 3000},
            )
            tile = _tile_response(tile_info)

            scl = collection.mosaic().select("SCL")
            cloud_mask = (
                scl.eq(3)
                .Or(scl.eq(8))
                .Or(scl.eq(9))
                .Or(scl.eq(10))
                .Or(scl.eq(11))
            )
            cloud_mean = cloud_mask.reduceRegion(
                reducer=_mean_reducer(),
                geometry=geometry,
                scale=20,
                bestEffort=True,
                maxPixels=1e13,
            ).get("SCL")
            if cloud_mean is not None:
                cloud_pct_value = float(ee.Number(cloud_mean).multiply(100).getInfo())

        cloud_values.append(cloud_pct_value)
        days.append(
            ImageryDayItem(
                date=day,
                tile=tile,
                cloud_pct=cloud_pct_value,
            )
        )

    avg_cloud = sum(cloud_values) / len(cloud_values) if cloud_values else 0.0
    summary = ImagerySummary(count=len(days), avg_cloud_pct=avg_cloud)
    return ImageryDailyResponse(days=days, summary=summary)


def _class_values(n_classes: int) -> List[int]:
    return list(range(1, n_classes + 1))


@router.post("/zones/basic", response_model=BasicZonesResponse)
@router.post("/products/zones/basic", response_model=BasicZonesResponse)
def zones_basic(request: BasicZonesRequest) -> BasicZonesResponse:
    if request.n_classes < 3 or request.n_classes > 9:
        raise HTTPException(status_code=400, detail="nClasses must be between 3 and 9.")
    if request.end < request.start:
        raise HTTPException(status_code=400, detail="end must be on or after start")

    mean_image = mean_ndvi(request.aoi, request.start, request.end)
    try:
        classification = classify_zones(
            mean_image,
            request.aoi,
            method="quantile",
            n_classes=request.n_classes,
            gaussian_radius_m=25,
            mode_radius_m=40,
            opening_radius_m=20,
            closing_radius_m=20,
            mmu_hectares=1.0,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(classification, Mapping):
        classes_img = classification.get("classes")
    else:
        classes_img = classification
    if classes_img is None:
        raise HTTPException(status_code=500, detail="Zone classification did not return classes image.")
    class_values = _class_values(request.n_classes)
    vectors = vectorize_zones(
        classes_img,
        request.aoi,
        simplify_tolerance_m=25,
        eight_connected=False,
        smooth_buffer_m=15,
        min_area_hectares=1.0,
    )
    stats_fc = _class_stats_fc(
        mean_image,
        classes_img,
        request.aoi,
        class_values=class_values,
    )

    try:
        raster_url = image_geotiff_url(
            classes_img, request.aoi, name="basic_zones_raster", scale=10
        )
        vector_url = table_shp_url(vectors, name="basic_zones_vectors")
        stats_url = table_csv_url(stats_fc, name="basic_zones_stats")
    except DownloadTooLargeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    palette = zone_palette(request.n_classes)

    tile_info = create_tile_session(
        classes_img,
        vis_params={
            "min": ZONES_VIS["min"],
            "max": request.n_classes,
            "palette": palette,
        },
    )

    try:
        vectors_geojson = vectors.getInfo()
    except Exception as exc:  # pragma: no cover - EE errors vary at runtime
        logger.exception("Failed to fetch basic zones vector overlay")
        raise HTTPException(status_code=502, detail="Failed to generate zones vector overlay.") from exc

    return BasicZonesResponse(
        preview={"tile": _tile_response(tile_info)},
        downloads=BasicDownloads(
            raster_geotiff=raster_url,
            vector_shp=vector_url,
            stats_csv=stats_url,
        ),
        vectors_geojson=vectors_geojson,
        class_count=request.n_classes,
    )


@router.post("/zones/advanced", response_model=AdvancedZonesResponse)
@router.post("/products/zones/advanced", response_model=AdvancedZonesResponse)
def zones_advanced(request: AdvancedZonesRequest) -> AdvancedZonesResponse:
    if len(request.breaks) != 4:
        raise HTTPException(status_code=400, detail="breaks must contain four values.")
    if not request.seasons:
        raise HTTPException(status_code=400, detail="At least one season is required.")

    season_defs = [
        SeasonDefinition(
            sowing_date=season.sowing_date,
            harvest_date=season.harvest_date,
        )
        for season in request.seasons
    ]

    layers = compute_advanced_layers(request.aoi, season_defs, request.breaks)
    composite_img = zones_img = vectors = dissolved_vectors = None
    if isinstance(layers, tuple):
        if len(layers) != 3:
            raise HTTPException(status_code=500, detail="Unexpected layers response.")
        composite_img, zones_img, vectors = layers
    else:
        composite_img = getattr(layers, "composite", None)
        zones_img = getattr(layers, "zones_raster", None) or getattr(layers, "zones", None)
        vectors = getattr(layers, "raw_zones", None) or getattr(layers, "vectors", None)
        dissolved_vectors = getattr(layers, "dissolved_zones", None)
    if composite_img is None or zones_img is None or vectors is None:
        raise HTTPException(status_code=500, detail="Advanced layers response missing required data.")
    if dissolved_vectors is None:
        dissolved_vectors = dissolve_by_class(vectors)
    n_classes = len(request.breaks) + 1
    class_values = _class_values(n_classes)
    stats_data = _attach_stats(
        composite_img,
        zones_img,
        request.aoi,
        dissolved_vectors=dissolved_vectors,
        band="Score",
        class_values=class_values,
    )
    if isinstance(stats_data, tuple):
        stats_raw, stats_dissolved = stats_data
    elif isinstance(stats_data, dict):
        stats_raw = stats_data.get("raw")
        stats_dissolved = stats_data.get("dissolved")
    else:
        stats_raw = stats_data
        stats_dissolved = stats_data

    try:
        raster_url = image_geotiff_url(
            zones_img, request.aoi, name="advanced_zones_raster", scale=10
        )
        vectors_url = table_shp_url(vectors, name="advanced_zones_vectors")
        vectors_dissolved_url = table_shp_url(
            dissolved_vectors, name="advanced_zones_vectors_dissolved"
        )
        stats_url = table_csv_url(stats_raw, name="advanced_zones_stats")
        stats_dissolved_url = table_csv_url(
            stats_dissolved, name="advanced_zones_stats_dissolved"
        )
    except DownloadTooLargeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    palette = zone_palette(n_classes)

    composite_tile = create_tile_session(
        composite_img,
        vis_params={"min": -2, "max": 2, "palette": ["440154", "21908d", "fde725"]},
    )
    zones_tile = create_tile_session(
        zones_img,
        vis_params={
            "min": ZONES_VIS["min"],
            "max": n_classes,
            "palette": palette,
        },
    )

    try:
        vectors_geojson = vectors.getInfo()
    except Exception as exc:  # pragma: no cover - EE errors vary at runtime
        logger.exception("Failed to fetch advanced zones vector overlay")
        raise HTTPException(status_code=502, detail="Failed to generate zones vector overlay.") from exc

    return AdvancedZonesResponse(
        preview=AdvancedPreview(
            composite={"tile": _tile_response(composite_tile)},
            zones={"tile": _tile_response(zones_tile)},
        ),
        downloads=AdvancedDownloads(
            zones_geotiff=raster_url,
            vectors_shp=vectors_url,
            vectors_dissolved_shp=vectors_dissolved_url,
            stats_csv=stats_url,
            stats_dissolved_csv=stats_dissolved_url,
        ),
        vectors_geojson=vectors_geojson,
        class_count=n_classes,
    )
