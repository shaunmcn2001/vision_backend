
from __future__ import annotations
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import ee

from app.services.tiles import get_tile_template_for_image, init_ee
from app.services.zones_workflow import monthly_ndvi, mean_ndvi, classify_zones, vectorize_zones
from app.services.advanced_zones import compute_advanced_layers
from app.services.export_drive import export_image_to_drive, export_table_to_drive

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
    paddock_name: str | None = None
    aoi: AOI
    start: str
    end: str
    bands: Optional[List[str]] = Field(default=["B4","B3","B2"])  # RGB

@router.post("/Imagery")
def imagery(req: ImageryRequest) -> Dict[str, Any]:
    init_ee()
    aoi = ee.Geometry(req.aoi.dict())
    start = ee.Date(req.start)
    end = ee.Date(req.end)
    days = end.difference(start, 'day').toInt()
    seq = ee.List.sequence(0, days)
    def per_day(i):
        d = ee.Date(start).advance(ee.Number(i), 'day')
        e = d.advance(1, 'day')
        col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(aoi).filterDate(d, e))
        # Natural color mosaic for the day (no cloud mask)
        img = col.mosaic().select(req.bands).clip(aoi)
        vis = img.visualize(min=0, max=3000)
        # Cloud percent via SCL (cloud/shadow/cirrus/snow)
        scl = col.mosaic().select('SCL')
        cloud = (scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10)).Or(scl.eq(11))).selfMask()
        cloud_pct = ee.Number(cloud.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi, scale=20, maxPixels=1e13, bestEffort=True
        ).get('SCL')).multiply(100)
        return ee.Feature(None, {'date': d.format('YYYY-MM-dd'), 'cloud_pct': cloud_pct, 'img': vis})
    feats = ee.FeatureCollection(seq.map(per_day))
    # Build tiles day-by-day (requires client loop to get map ids)
    items = []
    for f in feats.toList(feats.size()).getInfo():
        date = f['properties']['date']
        cloud_pct = f['properties'].get('cloud_pct', None)
        # rebuild image visualization from server-side object id? store not possible -> recompute quickly
        d = date
        col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(aoi).filterDate(d, ee.Date(d).advance(1,'day')))
        img = col.mosaic().select(req.bands).clip(aoi).visualize(min=0, max=3000)
        items.append({'date': date, 'cloud_pct': cloud_pct, **get_tile_template_for_image(img)})
    avg_cloud = None
    if items:
        vals = [x['cloud_pct'] for x in items if x['cloud_pct'] is not None]
        if vals:
            avg_cloud = sum(vals)/len(vals)
    return {"ok": True, "days": items, "summary": {"count": len(items), "avg_cloud_pct": avg_cloud}}

class BasicZonesRequest(BaseModel):
    aoi: AOI
    start: str
    end: str
    n_classes: int = 5
    paddock_name: str | None = None

@router.post("/Basic NDVI Zones")
def basic_zones(req: BasicZonesRequest) -> Dict[str, Any]:
    init_ee()
    aoi = ee.Geometry(req.aoi.dict())
    ndvi = mean_ndvi(aoi, req.start, req.end)
    out = classify_zones(ndvi, aoi, n_zones=req.n_classes, method='quantile', smooth_radius_m=0, mmu_pixels=0)
    classified = out["classes"]
    tile = get_tile_template_for_image(classified.visualize(min=1, max=req.n_classes, palette=['#440154','#31688e','#35b779','#fde725','#ffa600']))
    vectors = vectorize_zones(classified, aoi, simplify_tolerance_m=0.0)
    # Exports to Drive
    folder_root = "Vision Exports"
    product = "Basic NDVI Zones"
    raster_name = "basic_ndvi_zones"
    shp_name = "basic_ndvi_zones_vectors"
    # CRS: keep native (no explicit crs) or set to EPSG:32756? We'll omit -> EE picks native
    img_task = export_image_to_drive(classified.toInt(), aoi, raster_name, folder_root, req.paddock_name or "AOI", product, scale=10, crs=None, file_format="GeoTIFF")
    shp_task = export_table_to_drive(ee.FeatureCollection(vectors), shp_name, folder_root, req.paddock_name or "AOI", product, file_format="SHP")
    # Stats to CSV (Excel-friendly). Use your zones_workflow to compute stats if available; fallback simple area.
    # Here we export attributes already in vectors to CSV.
    csv_task = export_table_to_drive(ee.FeatureCollection(vectors), "basic_ndvi_zones_stats", folder_root, req.paddock_name or "AOI", product, file_format="CSV")
    return {"ok": True, "tile": tile, "export_tasks": {"raster": img_task, "shp": shp_task, "stats_csv": csv_task}}

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
    paddock_name: str | None = None

@router.post("/Advanced Zones")
def advanced_zones(req: AdvancedZonesRequest) -> Dict[str, Any]:
    init_ee()
    aoi = ee.Geometry(req.aoi.dict())
    lt_comp, lt_zones, vect = compute_advanced_layers(aoi, [r.dict() for r in req.seasons], req.breaks)
    tile_comp = get_tile_template_for_image(lt_comp.visualize(min=-2, max=2, palette=['#46039f','#1f9e89','#fde725']))
    tile_zones = get_tile_template_for_image(lt_zones.visualize(min=1, max=5, palette=['#440154','#31688e','#35b779','#fde725','#ffa600']))
    # Dissolve by ZONE
    zones_fc = ee.FeatureCollection(vect)
    def by_class(k):
        k = ee.Number(k)
        geom = zones_fc.filter(ee.Filter.eq('ZONE', k)).geometry().dissolve()
        return ee.Feature(geom, {'ZONE': k})
    dissolved = ee.FeatureCollection(ee.List.sequence(1,5).map(by_class))
    # Exports
    folder_root = "Vision Exports"
    product = "Advanced Zones"
    img_task = export_image_to_drive(lt_zones.toInt(), aoi, "advanced_zones_raster", folder_root, req.paddock_name or "AOI", product, scale=10, crs=None, file_format="GeoTIFF")
    shp_task = export_table_to_drive(zones_fc, "advanced_zones_vectors", folder_root, req.paddock_name or "AOI", product, file_format="SHP")
    shp_diss = export_table_to_drive(dissolved, "advanced_zones_dissolved", folder_root, req.paddock_name or "AOI", product, file_format="SHP")
    # Stats CSVs (per-polygon and dissolved)
    csv_stats = export_table_to_drive(zones_fc, "advanced_zones_stats", folder_root, req.paddock_name or "AOI", product, file_format="CSV")
    csv_diss  = export_table_to_drive(dissolved, "advanced_zones_dissolved_stats", folder_root, req.paddock_name or "AOI", product, file_format="CSV")
    return {"ok": True, "composite": tile_comp, "zones": tile_zones, "export_tasks": {"raster": img_task, "vectors": shp_task, "dissolved": shp_diss, "stats_csv": csv_stats, "dissolved_stats_csv": csv_diss}}


from fastapi import Query
from app.services.export_drive import task_status

@router.get("/export_status")
def export_status(task_id: str = Query(..., description="EE task id returned by an export call")):
    return {"task_id": task_id, **task_status(task_id)}
