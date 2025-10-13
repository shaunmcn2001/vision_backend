from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import date
from app.services import zones as zone_service

# Resolve defaults safely, so startup never crashes if zones.py is mid-refactor.
def _d(name: str, fallback):
    return getattr(zone_service, name, fallback)

DEFAULTS = {
    "DEFAULT_CLOUD_PROB_MAX": _d("DEFAULT_CLOUD_PROB_MAX", 100),
    "DEFAULT_N_CLASSES": _d("DEFAULT_N_CLASSES", 5),
    "DEFAULT_CV_THRESHOLD": _d("DEFAULT_CV_THRESHOLD", 0.8),
    "DEFAULT_MIN_MAPPING_UNIT_HA": _d("DEFAULT_MIN_MAPPING_UNIT_HA", 1.0),
    "DEFAULT_SMOOTH_RADIUS_M": _d("DEFAULT_SMOOTH_RADIUS_M", 0),
    "DEFAULT_OPEN_RADIUS_M": _d("DEFAULT_OPEN_RADIUS_M", 0),
    "DEFAULT_CLOSE_RADIUS_M": _d("DEFAULT_CLOSE_RADIUS_M", 0),
    "DEFAULT_SIMPLIFY_TOL_M": _d("DEFAULT_SIMPLIFY_TOL_M", 5),
    "DEFAULT_SIMPLIFY_BUFFER_M": _d("DEFAULT_SIMPLIFY_BUFFER_M", 3),
    "DEFAULT_METHOD": _d("DEFAULT_METHOD", "ndvi_percentiles"),
    "DEFAULT_MODE": _d("DEFAULT_MODE", "linear"),
}

class ProductionZoneOptions(BaseModel):
    aoi_geojson: Dict[str, Any]
    aoi_name: str = "aoi"
    months: List[str]

    # Core
    cloud_prob_max: int = Field(DEFAULTS["DEFAULT_CLOUD_PROB_MAX"], ge=0, le=100)
    n_classes: int = Field(DEFAULTS["DEFAULT_N_CLASSES"], ge=2, le=7)

    # UI extras
    zone_mode: str = Field(DEFAULTS["DEFAULT_MODE"], description="linear|quantile|auto")
    ndvi_min: Optional[float] = Field(None, description="Lower NDVI for linear bins")
    ndvi_max: Optional[float] = Field(None, description="Upper NDVI for linear bins")

    # Geometry & smoothing
    mmu_ha: float = Field(DEFAULTS["DEFAULT_MIN_MAPPING_UNIT_HA"], gt=0)
    smooth_radius_m: float = Field(DEFAULTS["DEFAULT_SMOOTH_RADIUS_M"], ge=0)

    # Simplification
    simplify_tolerance_m: float = Field(DEFAULTS["DEFAULT_SIMPLIFY_TOL_M"], ge=0)
    simplify_buffer_m: float = Field(DEFAULTS["DEFAULT_SIMPLIFY_BUFFER_M"], ge=0)

    # Extras
    include_stats: bool = True

router = APIRouter(prefix="/api/zones", tags=["zones"])

@router.post("/production")
def create_production_zones(opts: ProductionZoneOptions):
    try:
        result = zone_service.export_selected_period_zones(
            aoi_geojson=opts.aoi_geojson,
            aoi_name=opts.aoi_name,
            months=opts.months,
            cloud_prob_max=opts.cloud_prob_max,
            n_classes=opts.n_classes,
            min_mapping_unit_ha=opts.mmu_ha,
            smooth_radius_m=int(opts.smooth_radius_m),
            simplify_tolerance_m=int(opts.simplify_tolerance_m),
            simplify_buffer_m=int(opts.simplify_buffer_m),
            include_stats=opts.include_stats,
            mode=opts.zone_mode,
            ndvi_min=opts.ndvi_min,
            ndvi_max=opts.ndvi_max,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Zones failed: {e}")
