from __future__ import annotations
import ee
from datetime import date

S2_SR = "COPERNICUS/S2_SR_HARMONIZED"

def _mask_s2_sr_light(img: ee.Image) -> ee.Image:
    qa60 = img.select("QA60")
    cloud_mask = qa60.bitwiseAnd(1 << 10).eq(0).And(qa60.bitwiseAnd(1 << 11).eq(0))
    return img.updateMask(cloud_mask)

def _add_ndvi(img: ee.Image) -> ee.Image:
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI").toFloat()
    return img.addBands(ndvi)

def ndvi_mosaic(geom: ee.Geometry, start_date: date | str, end_date: date | str, reduce: str = "median") -> ee.Image:
    ic = (
        ee.ImageCollection(S2_SR)
        .filterDate(str(start_date), str(end_date))
        .filterBounds(geom)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
        .map(_mask_s2_sr_light)
        .map(_add_ndvi)
        .select("NDVI")
    )
    count = ic.size()
    def _empty():
        return ee.Image.constant(0).rename("NDVI").clip(geom)
    def _build():
        img = ee.Image(ee.Algorithms.If(reduce == "mean", ic.mean(), ic.median()))
        return img.toFloat().clip(geom)
    return ee.Image(ee.Algorithms.If(count.gt(0), _build(), _empty()))
