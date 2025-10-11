from __future__ import annotations
import ee
from app.services.ee_patches import apply_ee_runtime_patches
from app.services.ee_debug import debug_trace, debug_wrap  # noqa: F401

apply_ee_runtime_patches()


def normalize_ndvi_band(img: ee.Image) -> ee.Image:
    """
    Ensure a single band named 'NDVI' exists after reductions.
    Accepts 'NDVI' or 'NDVI_mean' and renames to 'NDVI'.
    """
    names = img.bandNames()
    has_ndvi = names.contains("NDVI")
    has_mean = names.contains("NDVI_mean")
    out = ee.Image(
        ee.Algorithms.If(
            has_ndvi,
            img.select(["NDVI"]).rename(["NDVI"]),
            ee.Algorithms.If(has_mean, img.select(["NDVI_mean"]).rename(["NDVI"]), img),
        )
    )
    return ee.Image(out).select(["NDVI"]).rename(["NDVI"])
