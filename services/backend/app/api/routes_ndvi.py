# services/backend/app/api/routes_ndvi.py
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List
import ee

from app.services import zones_workflow as zw
from app.services.ndvi_shared import reproject_native_10m

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
    export_crs: str = Field(
        "EPSG:3857",
        description="CRS for GeoTIFF exports (default Web Mercator, EPSG:3857)",
    )
    export_scale: float = 10.0


def _prepare_image_for_download(
    image: ee.Image,
    req: NDVIRequest,
    *,
    native_reference: ee.Image | None = None,
) -> ee.Image:
    """Return an image reprojected to the requested CRS/scale for downloads."""

    prepared = ee.Image(image)

    if native_reference is not None:
        try:
            prepared = ee.Image(
                reproject_native_10m(prepared, ee.Image(native_reference), ref_band="B8", scale=10)
            )
        except Exception:
            prepared = ee.Image(prepared)

    return prepared.reproject(req.export_crs, None, req.export_scale)

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
        lt_mean = zw.long_term_mean_ndvi(
            aoi,
            req.start,
            req.end,
            cloud_prob_max=req.cloud_prob_max,
        )

        z = zw.classify_zones(
            lt_mean, aoi,
            n_zones=req.n_zones,
            method=req.method,
            smooth_radius_m=req.smooth_radius_m,
            mmu_pixels=req.mmu_pixels
        )
        classified = z["classified"]
        breaks = z["breaks"]

        def img_url(img: ee.Image, name: str, *, native_ref: ee.Image | None = None) -> Dict[str, str]:
            params = {
                "name": name,
                "region": aoi,
                "crs": req.export_crs,
                "scale": req.export_scale,
                "filePerBand": False,
                "format": "GEO_TIFF",
            }
            download_img = _prepare_image_for_download(img, req, native_reference=native_ref)
            url = download_img.getDownloadURL(params)
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
        table_params_geojson = {"fileFormat": "geojson", "selectors": ["ZONE"], "region": aoi.toGeoJSONString()}
        table_params_kml = {"fileFormat": "kml", "selectors": ["ZONE"], "region": aoi.toGeoJSONString()}
        geojson_id = ee_data.getTableDownloadId(vectors, table_params_geojson)
        kml_id = ee_data.getTableDownloadId(vectors, table_params_kml)

        return {
            "monthly_ndvi": monthly_urls,
            "mean_ndvi": mean_url,
            "classified_zones": class_url,
            "zone_breaks": breaks.getInfo() if hasattr(breaks, "getInfo") else breaks,
            "vectors": {
                "geojson": ee_data.getTableDownloadUrl(geojson_id),
                "kml": ee_data.getTableDownloadUrl(kml_id),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NDVI generation failed: {e}")
