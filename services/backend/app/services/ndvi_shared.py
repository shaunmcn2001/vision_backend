from __future__ import annotations

import ee


DEFAULT_SCALE = 10  # metres


def compute_ndvi_loose(image: ee.Image, *, band_nir="B8", band_red="B4") -> ee.Image:
    """NDVI using upstream masks only; no extra clamps or dual-band mask."""

    return (
        image.select([band_nir, band_red])
        .toFloat()
        .normalizedDifference([band_nir, band_red])
        .rename("NDVI")
        .toFloat()
    )


def mean_from_collection_sum_count(ndvi_collection: ee.ImageCollection) -> ee.Image:
    """Mean NDVI = sum/count, masked where count==0."""

    summed = ndvi_collection.sum().rename("NDVI_sum")
    count = ndvi_collection.count().rename("NDVI_count")
    mean = summed.divide(count.max(1)).rename("NDVI_mean")
    return mean.updateMask(count.gt(0))


def reproject_native_10m(
    img: ee.Image, ref_image: ee.Image, *, ref_band="B8", scale=DEFAULT_SCALE
) -> ee.Image:
    """Reproject to the native projection of ref_image's ref_band (~10 m)."""

    proj = ref_image.select(ref_band).projection()
    return img.toFloat().resample("bilinear").reproject(proj, None, scale)
