# services/backend/app/services/zones_core.py
from __future__ import annotations
import ee
from app.services.ee_patches import apply_ee_runtime_patches
from app.services.ee_debug import debug_trace, debug_wrap  # noqa: F401
from .ndvi_helpers import normalize_ndvi_band
from .stability_mask import stability_mask_from_cv


apply_ee_runtime_patches()


__all__ = [
    "normalize_ndvi_band",
    "stability_mask_from_cv",
    "std_band",
    "coverage_ratio",
    "stats_stack",
]


@debug_wrap
def std_band(img: ee.Image) -> ee.Image:
    """
    Ensure a band named 'NDVI_stdDev' exists for CV.
    Accepts 'NDVI_stdDev' or falls back to 'NDVI' renamed.
    """
    names = img.bandNames()
    has_std = names.contains("NDVI_stdDev")
    has_ndvi = names.contains("NDVI")
    out = ee.Image(
        ee.Algorithms.If(
            has_std,
            img.select(["NDVI_stdDev"]).rename(["NDVI_stdDev"]),
            ee.Algorithms.If(
                has_ndvi, img.select(["NDVI"]).rename(["NDVI_stdDev"]), img
            ),
        )
    )
    return ee.Image(out).select(["NDVI_stdDev"]).rename(["NDVI_stdDev"])


# --- Coverage ---


@debug_wrap
def coverage_ratio(
    img: ee.Image, region: ee.Geometry, scale=10, tile_scale=4
) -> ee.Number:
    """Valid-pixel ratio (mask sum / total pixels)."""
    valid_sum = img.mask().reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=scale,
        bestEffort=True,
        maxPixels=1e9,
        tileScale=tile_scale,
    )
    total_px = region.area(1).divide(ee.Number(scale).pow(2))
    px_sum = ee.Number(valid_sum.values().reduce(ee.Reducer.sum()))
    return px_sum.divide(total_px)


# --- NDVI stats stack ---


@debug_wrap
def stats_stack(ndvi_ic: ee.ImageCollection) -> ee.Image:
    """
    Builds mean/stdDev/CV bands with consistent names:
      NDVI_mean, NDVI_stdDev, NDVI_cv
    """
    mean = ndvi_ic.mean().rename("NDVI_mean")
    stdv = ndvi_ic.reduce(ee.Reducer.stdDev()).rename("NDVI_stdDev")
    safe_mean = ee.Image(mean).where(ee.Image(mean).abs().lt(1e-6), 1e-6)
    cv = ee.Image(stdv).divide(safe_mean).rename("NDVI_cv")
    return mean.addBands(stdv).addBands(cv)
