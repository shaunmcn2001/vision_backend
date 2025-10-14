from __future__ import annotations
import ee
from datetime import date

S2_SR = "COPERNICUS/S2_SR_HARMONIZED"

def _add_ndvi(img: ee.Image) -> ee.Image:
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI").toFloat()
    return img.addBands(ndvi)

def _light_mask(ndvi_with_qa: ee.Image) -> ee.Image:
    qa60 = ndvi_with_qa.select("QA60")
    clear = qa60.bitwiseAnd(1 << 10).eq(0).And(qa60.bitwiseAnd(1 << 11).eq(0))
    return ndvi_with_qa.updateMask(clear)

def ndvi_mosaic(geom: ee.Geometry, start_date: date | str, end_date: date | str, reduce: str = "mean", apply_light_mask: bool = True) -> ee.Image:
    ic = (
        ee.ImageCollection(S2_SR)
        .filterDate(str(start_date), str(end_date))
        .filterBounds(geom)
        .select(["B4","B8","QA60"])
        .map(_add_ndvi)
    )
    count = ic.size()
    def _empty():
        return ee.Image.constant(0).rename("NDVI").clip(geom)
    def _build():
        ndvi_ic = ic.select("NDVI")
        ndvi = ee.Image(ndvi_ic.mean()) if reduce == "mean" else ee.Image(ndvi_ic.median())
        first = ee.Image(ic.first())
        ndvi = ndvi.addBands(first.select("QA60"))
        if apply_light_mask:
            ndvi = _light_mask(ndvi)
        return ndvi.select("NDVI").toFloat().clip(geom)
    return ee.Image(ee.Algorithms.If(count.gt(0), _build(), _empty()))
