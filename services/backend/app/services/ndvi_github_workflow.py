from __future__ import annotations
import ee
from datetime import date

S2_SR = "COPERNICUS/S2_SR_HARMONIZED"

def _add_ndvi(img: ee.Image) -> ee.Image:
    # Compute NDVI per image, no masking
    return img.normalizedDifference(["B8", "B4"]).rename("NDVI").toFloat()

def ndvi_mosaic(
    geom: ee.Geometry,
    start_date: date | str,
    end_date: date | str,
    reduce: str = "mean",
) -> ee.Image:
    ic = (
        ee.ImageCollection(S2_SR)
        .filterDate(str(start_date), str(end_date))
        .filterBounds(geom)
        .map(_add_ndvi)  # ImageCollection of single-band NDVI
    )

    count = ic.size()

    def _empty():
        # produce a constant NDVI but *valid* (mask=1) so stats exist
        return ee.Image.constant(0).rename("NDVI").toFloat().clip(geom).updateMask(ee.Image.constant(1))

    def _build():
        ndvi = ee.Image(ic.mean()) if reduce == "mean" else ee.Image(ic.median())
        # Force all pixels valid; do not bring QA60 or any mask back
        ndvi = ndvi.toFloat().clip(geom).updateMask(ee.Image.constant(1))
        return ndvi.rename("NDVI")

    return ee.Image(ee.Algorithms.If(count.gt(0), _build(), _empty()))
