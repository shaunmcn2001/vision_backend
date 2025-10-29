"""Pydantic schemas for Vision API payloads."""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _to_camel(value: str) -> str:
    parts = value.split("_")
    return parts[0] + "".join(part.capitalize() for part in parts[1:])


class CamelModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, alias_generator=_to_camel)


class TileResponse(CamelModel):
    token: str
    url_template: str = Field(alias="urlTemplate")
    min_zoom: int = Field(alias="minZoom")
    max_zoom: int = Field(alias="maxZoom")
    expires_at: Optional[str] = Field(default=None, alias="expiresAt")


class NDVIMonthRequest(CamelModel):
    aoi: Dict[str, Any]
    start: date
    end: date
    clamp: Optional[Sequence[float]] = None

    @field_validator("end")
    @classmethod
    def _validate_end(cls, value: date, info: Dict[str, Any]) -> date:
        start = info.data.get("start")
        if start and value < start:
            raise ValueError("end must be on or after start")
        return value

    @field_validator("clamp")
    @classmethod
    def _validate_clamp(cls, value: Optional[Sequence[float]]) -> Optional[Sequence[float]]:
        if value is not None and len(value) != 2:
            raise ValueError("clamp must contain two values [min, max]")
        return value


class NDVIMonthItem(CamelModel):
    name: str
    tile: TileResponse


class NDVIMonthResponse(CamelModel):
    items: List[NDVIMonthItem]
    mean: TileResponse


class ImageryDailyRequest(CamelModel):
    aoi: Dict[str, Any]
    start: date
    end: date
    bands: Optional[List[str]] = None

    @field_validator("end")
    @classmethod
    def _imagery_end(cls, value: date, info: Dict[str, Any]) -> date:
        start = info.data.get("start")
        if start and value < start:
            raise ValueError("end must be on or after start")
        return value


class ImageryDayItem(CamelModel):
    date: date
    tile: Optional[TileResponse] = None
    cloud_pct: float = Field(alias="cloudPct")


class ImagerySummary(CamelModel):
    count: int
    avg_cloud_pct: float = Field(alias="avgCloudPct")


class ImageryDailyResponse(CamelModel):
    days: List[ImageryDayItem]
    summary: ImagerySummary


class BasicZonesRequest(CamelModel):
    aoi: Dict[str, Any]
    start: date
    end: date
    n_classes: int = Field(default=5, alias="nClasses")

    @field_validator("end")
    @classmethod
    def _zones_end(cls, value: date, info: Dict[str, Any]) -> date:
        start = info.data.get("start")
        if start and value < start:
            raise ValueError("end must be on or after start")
        return value


class BasicDownloads(CamelModel):
    raster_geotiff: str = Field(alias="rasterGeotiff")
    vector_shp: str = Field(alias="vectorShp")
    stats_csv: str = Field(alias="statsCsv")


class TileWrapper(CamelModel):
    tile: TileResponse


class BasicZonesResponse(CamelModel):
    preview: TileWrapper
    downloads: BasicDownloads


class SeasonInput(CamelModel):
    field_name: Optional[str] = Field(default=None, alias="fieldName")
    field_id: Optional[str] = Field(default=None, alias="fieldId")
    crop: Optional[str] = None
    sowing_date: date = Field(alias="sowingDate")
    harvest_date: date = Field(alias="harvestDate")
    emergence_date: Optional[date] = Field(default=None, alias="emergenceDate")
    flowering_date: Optional[date] = Field(default=None, alias="floweringDate")
    yield_asset: Optional[str] = Field(default=None, alias="yieldAsset")
    soil_asset: Optional[str] = Field(default=None, alias="soilAsset")

    @field_validator("harvest_date")
    @classmethod
    def _harvest_after_sowing(cls, value: date, info: Dict[str, Any]) -> date:
        sowing = info.data.get("sowing_date")
        if sowing and value <= sowing:
            raise ValueError("harvest_date must be after sowing_date")
        return value


class AdvancedZonesRequest(CamelModel):
    aoi: Dict[str, Any]
    breaks: Sequence[float]
    seasons: List[SeasonInput]


class AdvancedDownloads(CamelModel):
    zones_geotiff: str = Field(alias="zonesGeotiff")
    vectors_shp: str = Field(alias="vectorsShp")
    vectors_dissolved_shp: str = Field(alias="vectorsDissolvedShp")
    stats_csv: str = Field(alias="statsCsv")
    stats_dissolved_csv: str = Field(alias="statsDissolvedCsv")


class AdvancedPreview(CamelModel):
    composite: TileWrapper
    zones: TileWrapper


class AdvancedZonesResponse(CamelModel):
    preview: AdvancedPreview
    downloads: AdvancedDownloads


class TileSessionRequest(CamelModel):
    image: str
    vis_params: Optional[Dict[str, Any]] = Field(default=None, alias="visParams")
    min_zoom: int = Field(default=6, alias="minZoom")
    max_zoom: int = Field(default=18, alias="maxZoom")

    @field_validator("max_zoom")
    @classmethod
    def _max_not_less_than_min(cls, value: int, info: Dict[str, Any]) -> int:
        min_zoom = info.data.get("min_zoom")
        if min_zoom is not None and value < min_zoom:
            raise ValueError("max_zoom must be greater than or equal to min_zoom")
        return value


class TileSessionResponse(CamelModel):
    token: str
    url_template: str = Field(alias="urlTemplate")
    min_zoom: int = Field(alias="minZoom")
    max_zoom: int = Field(alias="maxZoom")
    expires_at: Optional[str] = Field(default=None, alias="expiresAt")
