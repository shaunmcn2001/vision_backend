
from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from typing import Optional, Dict, Any
import json, io, zipfile, os

from app.services.tiles import init_ee
from app.api.products import ndvi_month, imagery, basic_zones
from pydantic import BaseModel
import shapefile  # pyshp; if missing, instruct user to upload GeoJSON

router = APIRouter(tags=["ui"])

def _shapefile_zip_to_geojson_bytes(zf: zipfile.ZipFile) -> bytes:
    # Find core parts
    names = {n.lower(): n for n in zf.namelist()}
    shp = next((names[n] for n in names if n.endswith('.shp')), None)
    shx = next((names[n] for n in names if n.endswith('.shx')), None)
    dbf = next((names[n] for n in names if n.endswith('.dbf')), None)
    if not (shp and shx and dbf):
        raise HTTPException(400, "ZIP must contain .shp, .shx, .dbf")
    # Read with pyshp
    r = shapefile.Reader(shp=io.BytesIO(zf.read(shp)),
                         shx=io.BytesIO(zf.read(shx)),
                         dbf=io.BytesIO(zf.read(dbf)))
    # Build a dissolved MultiPolygon/MultiLine AOI (use first polygon if multiple)
    # Prefer polygon layer
    geoms = []
    for s in r.shapes():
        if s.shapeType in (5,15,25):  # POLYGON types
            geoms.append(s.__geo_interface__)
    if not geoms:
        raise HTTPException(400, "No polygon geometry found in shapefile.")
    # Use the first polygon as AOI to keep it simple
    aoi = geoms[0]
    # Ensure Feature-like object
    feat = {"type":"Feature", "geometry": aoi, "properties": {}}
    fc = {"type":"FeatureCollection", "features":[feat]}
    return json.dumps(fc).encode("utf-8")

@router.get("/ui", response_class=HTMLResponse)
def form(request: Request):
    html = (request.app.state.templates_env.get_template("ui.html")).render()
    return HTMLResponse(html)

@router.post("/ui/run")
async def ui_run(
    product: str = Form(..., description="One of: NDVI Month, Imagery, Basic NDVI Zones"),
    start: str = Form(...),
    end: str = Form(...),
    paddock_name: Optional[str] = Form(None),
    aoi_geojson: Optional[str] = Form(None),
    shp_zip: Optional[UploadFile] = File(None)
):
    # Parse AOI either from GeoJSON text or uploaded shapefile zip
    if aoi_geojson:
        try:
            gj = json.loads(aoi_geojson)
        except Exception as e:
            raise HTTPException(400, f"Bad GeoJSON: {e}")
        # Expect FeatureCollection or Polygon Feature
        if gj.get("type") == "FeatureCollection":
            geom = gj["features"][0]["geometry"]
        elif gj.get("type") == "Feature":
            geom = gj["geometry"]
        else:
            geom = gj
    elif shp_zip:
        content = await shp_zip.read()
        try:
            zf = zipfile.ZipFile(io.BytesIO(content), "r")
        except zipfile.BadZipFile:
            raise HTTPException(400, "Upload must be a .zip of the Shapefile.")
        gj_bytes = _shapefile_zip_to_geojson_bytes(zf)
        fc = json.loads(gj_bytes.decode("utf-8"))
        geom = fc["features"][0]["geometry"]
    else:
        raise HTTPException(400, "Provide either a GeoJSON AOI or upload a Shapefile .zip")

    # Dispatch to product endpoints
    if product == "NDVI Month":
        body = {"aoi": geom, "start": start, "end": end, "clamp":[0.0,1.0]}
        res = ndvi_month.__wrapped__(ndvi_month)(body) if hasattr(ndvi_month, "__wrapped__") else ndvi_month(body)  # if FastAPI wrappers
        return JSONResponse(res)
    elif product == "Imagery":
        body = {"aoi": geom, "start": start, "end": end, "bands":["B4","B3","B2"], "paddock_name": paddock_name}
        res = imagery.__wrapped__(imagery)(body) if hasattr(imagery, "__wrapped__") else imagery(body)
        return JSONResponse(res)
    elif product == "Basic NDVI Zones":
        body = {"aoi": geom, "start": start, "end": end, "n_classes":5, "paddock_name": paddock_name}
        res = basic_zones.__wrapped__(basic_zones)(body) if hasattr(basic_zones, "__wrapped__") else basic_zones(body)
        return JSONResponse(res)
    else:
        raise HTTPException(400, "Unsupported product for UI. Advanced Zones requires seasons CSV and is not in this simple form yet.")
