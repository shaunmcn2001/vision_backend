# services/backend/app/services/zones_core.py
from __future__ import annotations
import ee
from .ee_utils import ensure_list

# --- Band normalization ---

def normalize_ndvi_band(img: ee.Image) -> ee.Image:
    """
    Ensure a single band named 'NDVI' exists after reductions.
    Accepts 'NDVI' or 'NDVI_mean' and renames to 'NDVI'.
    """
    names = img.bandNames()
    has_ndvi = names.contains("NDVI")
    has_mean = names.contains("NDVI_mean")
    out = ee.Image(ee.Algorithms.If(
        has_ndvi, img.select(["NDVI"]).rename(["NDVI"]),
        ee.Algorithms.If(has_mean, img.select(["NDVI_mean"]).rename(["NDVI"]), img)
    ))
    return ee.Image(out).select(["NDVI"]).rename(["NDVI"])

def std_band(img: ee.Image) -> ee.Image:
    """
    Ensure a band named 'NDVI_stdDev' exists for CV.
    Accepts 'NDVI_stdDev' or falls back to 'NDVI' renamed.
    """
    names = img.bandNames()
    has_std = names.contains("NDVI_stdDev")
    has_ndvi = names.contains("NDVI")
    out = ee.Image(ee.Algorithms.If(
        has_std, img.select(["NDVI_stdDev"]).rename(["NDVI_stdDev"]),
        ee.Algorithms.If(has_ndvi, img.select(["NDVI"]).rename(["NDVI_stdDev"]), img)
    ))
    return ee.Image(out).select(["NDVI_stdDev"]).rename(["NDVI_stdDev"])

# --- Coverage ---

def coverage_ratio(img: ee.Image, region: ee.Geometry, scale=10, tile_scale=4) -> ee.Number:
    """Valid-pixel ratio (mask sum / total pixels)."""
    valid_sum = img.mask().reduceRegion(
        reducer=ee.Reducer.sum(), geometry=region, scale=scale,
        bestEffort=True, maxPixels=1e9, tileScale=tile_scale
    )
    total_px = region.area(1).divide(ee.Number(scale).pow(2))
    px_sum = ee.Number(valid_sum.values().reduce(ee.Reducer.sum()))
    return px_sum.divide(total_px)

# --- NDVI stats stack ---

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

# --- Stability mask (GEE JS parity, Images only) ---

def stability_mask_from_cv(
    cv_image: ee.Image,
    region: ee.Geometry,
    thresholds,                  # scalar or list; we normalize
    scale=10,
    tile_scale=4,
    min_survival_ratio=0.0
) -> ee.Image:
    """
    For each threshold t in thresholds:
      m = (cv <= t), keep m only if surviving/total >= min_survival_ratio
    Combine via ImageCollection.max().
    If total==0 -> pass-through (1), to avoid nuking small AOIs.
    """
    total = ee.Number(cv_image.reduceRegion(
        reducer=ee.Reducer.count(), geometry=region, scale=scale,
        bestEffort=True, maxPixels=1e9, tileScale=tile_scale
    ).values().get(0))

    tlist = ensure_list(thresholds)

    def _one(t):
        t = ee.Number(t)
        m = cv_image.lte(t)
        surviving = ee.Number(m.reduceRegion(
            reducer=ee.Reducer.count(), geometry=region, scale=scale,
            bestEffort=True, maxPixels=1e9, tileScale=tile_scale
        ).values().get(0))
        ratio = surviving.divide(total.max(1))
        # IMPORTANT: return an Image on all branches
        return ee.Image(ee.Algorithms.If(ratio.gte(min_survival_ratio), m, ee.Image(0)))

    masks = tlist.map(_one)  # list of Images guaranteed
    combined = ee.ImageCollection.fromImages(masks).max()
    pass_through = ee.Image(1)
    return ee.Image(ee.Algorithms.If(total.lte(0), pass_through, combined)).selfMask()
