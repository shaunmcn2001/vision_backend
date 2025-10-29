"""Product endpoints for Vision backend."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, Iterable, List

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
    mean_ndvi,
    monthly_ndvi,
    vectorize_zones,
    zone_statistics,
)

router = APIRouter()

RGB_BANDS = ["B4", "B3", "B2"]


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


@router.post("/products/ndvi-month", response_model=NDVIMonthResponse)
def ndvi_month(request: NDVIMonthRequest) -> NDVIMonthResponse:
    if request.end < request.start:
        raise HTTPException(status_code=400, detail="end must be on or after start")

    months = _month_sequence(request.start, request.end)
    collection = monthly_ndvi(request.aoi, request.start, request.end)
    collection_size = int(collection.size().getInfo())
    image_list = collection.toList(collection_size)
    vis_params = dict(NDVI_VIS)
    if request.clamp:
        vis_params["min"] = float(request.clamp[0])
        vis_params["max"] = float(request.clamp[1])

    items: List[NDVIMonthItem] = []
    for idx, month_start in enumerate(months):
        if idx >= collection_size:
            break
        image = ee.Image(image_list.get(idx))
        label = f"ndvi_{month_start.strftime('%Y-%m')}"
        tile_info = create_tile_session(image, vis_params=vis_params)
        items.append(
            NDVIMonthItem(
                name=label,
                tile=_tile_response(tile_info),
            )
        )

    mean_image = mean_ndvi(request.aoi, request.start, request.end)
    mean_tile = create_tile_session(mean_image, vis_params=vis_params)
    return NDVIMonthResponse(items=items, mean=_tile_response(mean_tile))


def _iterate_days(start: date, end: date) -> Iterable[date]:
    cursor = start
    while cursor <= end:
        yield cursor
        cursor += timedelta(days=1)


@router.post("/products/imagery/daily", response_model=ImageryDailyResponse)
def imagery_daily(request: ImageryDailyRequest) -> ImageryDailyResponse:
    if request.end < request.start:
        raise HTTPException(status_code=400, detail="end must be on or after start")

    ensure_ee()
    bands = request.bands or RGB_BANDS
    geometry = to_ee_geometry(request.aoi)
    days: List[ImageryDayItem] = []
    cloud_values: List[float] = []

    for day in _iterate_days(request.start, request.end):
        day_start = ee.Date(day.isoformat())
        day_end = ee.Date((day + timedelta(days=1)).isoformat())
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate(day_start, day_end)
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
                reducer=ee.Reducer.mean(),
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


@router.post("/products/zones/basic", response_model=BasicZonesResponse)
def zones_basic(request: BasicZonesRequest) -> BasicZonesResponse:
    if request.n_classes != 5:
        raise HTTPException(status_code=400, detail="Only 5-class zones are supported.")
    if request.end < request.start:
        raise HTTPException(status_code=400, detail="end must be on or after start")

    mean_image = mean_ndvi(request.aoi, request.start, request.end)
    classification = classify_zones(mean_image, request.aoi, method="quantile")
    classes_img = classification["classes"]
    class_values = _class_values(request.n_classes)
    vectors = vectorize_zones(classes_img, request.aoi)
    stats_fc = zone_statistics(
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

    tile_info = create_tile_session(
        classes_img,
        vis_params={
            "min": ZONES_VIS["min"],
            "max": ZONES_VIS["max"],
            "palette": ZONES_VIS["palette"],
        },
    )

    return BasicZonesResponse(
        preview={"tile": _tile_response(tile_info)},
        downloads=BasicDownloads(
            raster_geotiff=raster_url,
            vector_shp=vector_url,
            stats_csv=stats_url,
        ),
    )


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

    composite_img, zones_img, vectors = compute_advanced_layers(
        request.aoi, season_defs, request.breaks
    )
    dissolved_vectors = dissolve_by_class(vectors)
    class_values = _class_values(5)
    stats_raw = zone_statistics(
        composite_img,
        zones_img,
        request.aoi,
        band="Score",
        class_values=class_values,
    )
    stats_dissolved = dissolved_zone_statistics(
        composite_img,
        zones_img,
        dissolved_vectors,
        band="Score",
    )

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

    composite_tile = create_tile_session(
        composite_img,
        vis_params={"min": -2, "max": 2, "palette": ["440154", "21908d", "fde725"]},
    )
    zones_tile = create_tile_session(
        zones_img,
        vis_params={
            "min": ZONES_VIS["min"],
            "max": ZONES_VIS["max"],
            "palette": ZONES_VIS["palette"],
        },
    )

    return AdvancedZonesResponse(
        preview=AdvancedPreview(
            composite=_tile_response(composite_tile),
            zones=_tile_response(zones_tile),
        ),
        downloads=AdvancedDownloads(
            zones_geotiff=raster_url,
            vectors_shp=vectors_url,
            vectors_dissolved_shp=vectors_dissolved_url,
            stats_csv=stats_url,
            stats_dissolved_csv=stats_dissolved_url,
        ),
    )
