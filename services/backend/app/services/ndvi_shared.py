from __future__ import annotations

import ee

DEFAULT_SCALE = 10  # meters


def compute_ndvi_loose(image: ee.Image, *, band_nir: str = "B8", band_red: str = "B4") -> ee.Image:
    """
    NDVI with the same less-strict behavior as the general path:
      - NDVI = normalizedDifference([NIR, RED]) renamed 'NDVI'
      - No extra both-band mask
      - No NDVI < 1 clamp
      - Relies entirely on upstream image mask (cloud/shadow/etc.)
    """
    bands = image.select([band_nir, band_red]).toFloat()
    return bands.normalizedDifference([band_nir, band_red]).rename("NDVI").toFloat()


def mean_from_collection_sum_count(ndvi_collection: ee.ImageCollection) -> ee.Image:
    """
    Mean NDVI via sum/count on masked NDVI images.
    Returns single band 'NDVI_mean' masked to where count > 0.
    """
    summed = ndvi_collection.sum().rename("NDVI_sum")
    count = ndvi_collection.count().rename("NDVI_count")
    mean = summed.divide(count.max(1)).rename("NDVI_mean")
    return mean.toFloat().updateMask(count.gt(0))


def reproject_native_10m(
    img: ee.Image, ref_image: ee.Image, *, ref_band: str = "B8", scale: int = DEFAULT_SCALE
) -> ee.Image:
    """
    Reproject 'img' to the native projection of ref_image's ref_band at ~10 m for analysis steps.
    """
    proj = ref_image.select(ref_band).projection()
    return img.resample("bilinear").reproject(proj, None, scale)
