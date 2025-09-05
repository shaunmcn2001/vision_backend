# services/backend/app/api/routes.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from ..services import earth_engine as gee

router = APIRouter()

# -------------------
# Basic checks
# -------------------

@router.get("/ping")
def ping():
    """Simple ping endpoint."""
    return {"message": "pong"}

@router.get("/healthz")
def healthz():
    """Health check endpoint to verify container is up."""
    return {"ok": True, "service": "vision-backend"}

# -------------------
# Earth Engine health
# -------------------

@router.get("/ee/health")
def ee_health():
    """Check if Earth Engine initialization works."""
    try:
        result = gee.ee_ping()
        return {"ok": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------
# NDVI Monthly computation
# -------------------

class NDVIRequest(BaseModel):
    geometry: Dict[str, Any]          # GeoJSON Polygon or Feature
    start: str = "2019-01-01"
    end: str = "2025-01-01"

@router.post("/ndvi/monthly")
def ndvi_monthly(req: NDVIRequest):
    """
    Compute monthly mean NDVI for a given geometry between start and end dates.
    Example geometry:
    {
      "type": "Polygon",
      "coordinates": [[[149.6,-31.55],[149.7,-31.55],[149.7,-31.45],[149.6,-31.45],[149.6,-31.55]]]
    }
    """
    try:
        gee.ensure_ee()
        import ee
        geom = ee.Geometry(req.geometry) if req.geometry.get('type') != 'Feature' \
               else ee.Geometry(req.geometry['geometry'])

        s2 = (ee.ImageCollection('COPERNICUS/S2_SR')
              .filterBounds(geom)
              .filterDate(req.start, req.end)
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))
              .map(lambda img: img.updateMask(img.select('SCL').neq(9)))  # mask snow/shadows
        )

        def add_ndvi(img):
            ndvi = img.normalizedDifference(['B8','B4']).rename('NDVI')
            return img.addBands(ndvi)

        s2 = s2.map(add_ndvi)

        # Monthly composites
        def by_month(y, m):
            start = ee.Date.fromYMD(y, m, 1)
            end   = start.advance(1, 'month')
            ndvi_mean = s2.filterDate(start, end).select('NDVI').mean()
            stat = ndvi_mean.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geom, scale=10, bestEffort=True
            )
            return ee.Feature(None, {
                'date': start.format('YYYY-MM').getInfo(),
                'ndvi_mean': stat.get('NDVI')
            })

        start = ee.Date(req.start)
        end   = ee.Date(req.end)
        months = end.difference(start, 'month').floor().getInfo()

        feats = []
        for i in range(int(months) + 1):
            d = start.advance(i, 'month')
            yyyy = int(d.format('Y').getInfo())
            mm   = int(d.format('M').getInfo())
            feats.append(by_month(yyyy, mm))

        res = [f.getInfo()['properties'] for f in feats]
        return {"series": res}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
