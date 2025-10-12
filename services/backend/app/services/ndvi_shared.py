from __future__ import annotations

import ee

DEFAULT_SCALE = 10  # meters


def compute_ndvi_loose(image: ee.Image, *, band_nir: str = "B8", band_red: str = "B4") -> ee.Image:
    """
    NDVI matching the general path: normalizedDifference([B8, B4]), float, no extra masks/clamps.
    Relies entirely on the upstream mask (cloud/shadow, etc.).
    """
    bands = image.select([band_nir, band_red]).toFloat()
    return bands.normalizedDifference([band_nir, band_red]).rename("NDVI").toFloat()


def mean_from_collection_sum_count(ndvi_collection: ee.ImageCollection) -> ee.Image:
    """
    Mean NDVI via sum/count using masked NDVI images. Outputs single band 'NDVI_mean' masked where count==0.
    """
    summed = ndvi_collection.sum().rename("NDVI_sum")
    count = ndvi_collection.count().rename("NDVI_count")
    mean = summed.divide(count.max(1)).rename("NDVI_mean")
    return mean.updateMask(count.gt(0))


def reproject_native_10m(
    img: ee.Image, ref_image: ee.Image, *, ref_band: str = "B8", scale: int = DEFAULT_SCALE
) -> ee.Image:
    """
    Reproject 'img' to the native projection of ref_image's ref_band at ~10 m for analysis steps.
    """
    proj = ref_image.select(ref_band).projection()
    return img.toFloat().resample("bilinear").reproject(proj, None, scale)
