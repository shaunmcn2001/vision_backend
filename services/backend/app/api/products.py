
from __future__ import annotations
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import ee

from app.services.tiles import get_tile_template_for_image, init_ee
from app.services.zones_workflow import monthly_ndvi, mean_ndvi, classify_zones, vectorize_zones
from app.services.advanced_zones import compute_advanced_layers

router = APIRouter(prefix="/api", tags=["products"])

class AOI(BaseModel):
    type: str
    coordinates: List[List[List[float]]]

class NDVIMonthRequest(BaseModel):
    aoi: AOI
    start: str  # 'YYYY-MM-01'
    end: str    # 'YYYY-MM-31' or last day
    clamp: Optional[List[float]] = Field(default=[0.0, 1.0])

@router.post("/NDVI Month")
def ndvi_month(req: NDVIMonthRequest) -> Dict[str, Any]:
    init_ee()
    aoi = ee.Geometry(req.aoi.dict())
    monthly = monthly_ndvi(aoi, req.start, req.end)
    if monthly.size().getInfo() == 0:
        raise HTTPException(400, "No imagery for requested month/date range.")
    # Return tiles for each monthly image and the mean
    monthly_list = monthly.toList(monthly.size())
    items: List[Dict[str, Any]] = []
    for i in range(int(monthly.size().getInfo())):
        img = ee.Image(monthly_list.get(i)).select("NDVI")
        props = img.toDictionary()
        y = ee.Number(props.get("year")).format().getInfo()
        m = ee.Number(props.get("month")).format("02d").getInfo()
        name = f"ndvi_{y}-{m}"
        tile = get_tile_template_for_image(img.visualize(min=req.clamp[0], max=req.clamp[1], palette=['white','green']))
        items.append({"name": name, **tile})
    mean_img = mean_ndvi(aoi, req.start, req.end).select("NDVI")
    mean_tile = get_tile_template_for_image(mean_img.visualize(min=req.clamp[0], max=req.clamp[1], palette=['white','green']))
    return {"ok": True, "items": items, "mean": mean_tile}

class ImageryRequest(BaseModel):
    aoi: AOI
    start: str
    end: str
    bands: Optional[List[str]] = Field(default=["B4","B3","B2"])  # RGB

@router.post("/Imagery")
def imagery(req: ImageryRequest) -> Dict[str, Any]:
    # Placeholder: return a simple natural color composite tile template
    init_ee()
    aoi = ee.Geometry(req.aoi.dict())
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(aoi).filterDate(req.start, req.end)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40)))
    if col.size().getInfo() == 0:
        raise HTTPException(400, "No Sentinel-2 scenes in that window.")
    img = col.sort("CLOUDY_PIXEL_PERCENTAGE").first().select(req.bands).clip(aoi)
    vis = img.visualize(min=0, max=3000)
    tile = get_tile_template_for_image(vis)
    return {"ok": True, "tile": tile}

class BasicZonesRequest(BaseModel):
    aoi: AOI
    start: str
    end: str
    n_classes: int = 5

@router.post("/Basic NDVI Zones")
def basic_zones(req: BasicZonesRequest) -> Dict[str, Any]:
    init_ee()
    aoi = ee.Geometry(req.aoi.dict())
    ndvi = mean_ndvi(aoi, req.start, req.end)
    classified = classify_zones(ndvi, aoi, n_zones=req.n_classes, method='quantile', smooth_radius_m=0, mmu_pixels=0)["classes"]
    tile = get_tile_template_for_image(classified.visualize(min=1, max=req.n_classes, palette=['#440154','#31688e','#35b779','#fde725','#ffa600']))
    vectors = vectorize_zones(classified, aoi, simplify_tolerance_m=0.0)
    return {"ok": True, "tile": tile, "vectors_geojson": vectors.getInfo()}  # small AOIs only

class SeasonRow(BaseModel):
    field_name: Optional[str] = None
    field_id: Optional[str] = None
    crop: str
    sowing_date: str
    harvest_date: str
    yield_asset: Optional[str] = None
    soil_asset: Optional[str] = None

class AdvancedZonesRequest(BaseModel):
    aoi: AOI
    seasons: List[SeasonRow]
    breaks: List[float] = Field(default=[-1.0, -0.3, 0.3, 1.0])

@router.post("/Advanced Zones")
def advanced_zones(req: AdvancedZonesRequest) -> Dict[str, Any]:
    init_ee()
    aoi = ee.Geometry(req.aoi.dict())
    lt_comp, lt_zones, vect = compute_advanced_layers(aoi, [r.dict() for r in req.seasons], req.breaks)
    tile_comp = get_tile_template_for_image(lt_comp.visualize(min=-2, max=2, palette=['#46039f','#1f9e89','#fde725']))
    tile_zones = get_tile_template_for_image(lt_zones.visualize(min=1, max=5, palette=['#440154','#31688e','#35b779','#fde725','#ffa600']))
    return {"ok": True, "composite": tile_comp, "zones": tile_zones, "vectors_geojson": vect.getInfo()}
