
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import ee

from app.services import zones_workflow as zw

router = APIRouter(prefix="/api/zones", tags=["zones"])

class AOI(BaseModel):
    type: str = "Polygon"
    coordinates: List[List[List[float]]]

class NDVIRequest(BaseModel):
    aoi: AOI
    start: str = Field(..., description="YYYY-MM-DD")
    end: str = Field(..., description="YYYY-MM-DD")
    cloud_prob_max: int = 40
    mask_scl_shadow_snow: bool = True
    n_zones: int = 5
    method: str = "quantile"
    smooth_radius_m: int = 30
    mmu_pixels: int = 50
    export_crs: str = "EPSG:4326"
    export_scale: float = 10.0

@router.post("/ndvi", summary="Generate NDVI artifacts and return download URLs")
def generate_ndvi(req: NDVIRequest) -> Dict[str, Any]:
    try:
        zw.init_ee()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"EE init failed: {e}")

    try:
        aoi = ee.Geometry(req.aoi.dict())
        s2 = zw.get_s2_sr_collection(
            aoi, req.start, req.end,
            cloud_prob_max=req.cloud_prob_max,
            mask_scl_shadow_snow=req.mask_scl_shadow_snow
        )

        monthly = zw.monthly_ndvi_mean(s2)
        lt_mean = zw.long_term_mean_ndvi(s2)

        z = zw.classify_zones(
            lt_mean, aoi,
            n_zones=req.n_zones,
            method=req.method,
            smooth_radius_m=req.smooth_radius_m,
            mmu_pixels=req.mmu_pixels
        )
        classified = z["classified"]
        breaks = z["breaks"]

        def img_url(img: ee.Image, name: str) -> Dict[str, str]:
            params = {
                "name": name,
                "region": aoi,
                "crs": req.export_crs,
                "scale": req.export_scale,
                "filePerBand": False,
                "format": "GEO_TIFF"
            }
            url = img.getDownloadURL(params)
            return {"name": name, "url": url}

        monthly_list = monthly.toList(monthly.size())
        monthly_urls = []
        size = monthly.size().getInfo()
        for i in range(size):
            img = ee.Image(monthly_list.get(i))
            y = ee.Number(img.get("year")).format().getInfo() if img.get("year") else "YYYY"
            m = ee.Number(img.get("month")).format("02d").getInfo() if img.get("month") else "MM"
            name = f"ndvi_{y}-{m}"
            try:
                monthly_urls.append(img_url(img, name))
            except Exception:
                continue

        mean_url = img_url(lt_mean, "ndvi_mean")
        class_url = img_url(classified.toInt(), "ndvi_zones_5")

        vectors = zw.vectorize_zones(classified, aoi, simplify_tolerance_m=5.0)

        from ee import data as ee_data
        table_params_geojson = {
            "fileFormat": "geojson",
            "selectors": ["ZONE"],
            "region": aoi.toGeoJSONString()
        }
        table_params_kml = {
            "fileFormat": "kml",
            "selectors": ["ZONE"],
            "region": aoi.toGeoJSONString()
        }
        geojson_id = ee_data.getTableDownloadId(vectors, table_params_geojson)
        kml_id = ee_data.getTableDownloadId(vectors, table_params_kml)
        geojson_url = ee_data.getTableDownloadUrl(geojson_id)
        kml_url = ee_data.getTableDownloadUrl(kml_id)

        return {
            "monthly_ndvi": monthly_urls,
            "mean_ndvi": mean_url,
            "classified_zones": class_url,
            "zone_breaks": breaks.getInfo() if hasattr(breaks, "getInfo") else breaks,
            "vectors": {
                "geojson": geojson_url,
                "kml": kml_url
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NDVI generation failed: {e}")
