
from __future__ import annotations
import ee
from datetime import date

S2_SR = 'COPERNICUS/S2_SR_HARMONIZED'

def _mask_s2_sr(img: ee.Image) -> ee.Image:
    qa60 = img.select('QA60')
    cloud_bits_clear = qa60.bitwiseAnd(1 << 10).eq(0).And(qa60.bitwiseAnd(1 << 11).eq(0))
    scl = img.select('SCL')
    good_scl = (scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7)))
    mask = cloud_bits_clear.And(good_scl)
    return img.updateMask(mask)

def _ndvi(img: ee.Image) -> ee.Image:
    ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI').toFloat()
    valid = img.select('B8').mask().And(img.select('B4').mask())
    return ndvi.updateMask(valid)

def ndvi_mosaic(geom: ee.Geometry, start_date: date | str, end_date: date | str, reduce: str = 'median') -> ee.Image:
    start = str(start_date)
    end = str(end_date)
    ic = (ee.ImageCollection(S2_SR)
            .filterDate(start, end)
            .filterBounds(geom)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70))
            .map(_mask_s2_sr)
            .map(lambda i: i.toFloat()))
    count = ic.size()

    def _empty():
        return ee.Image(0).rename('NDVI').updateMask(ee.Image(0))

    def _build():
        ndvi_ic = ic.map(_ndvi).select('NDVI')
        img = ee.Image(ee.Algorithms.If(reduce == 'mean', ndvi_ic.mean(), ndvi_ic.median()))
        first = ee.Image(ic.first())
        proj = first.select('B8').projection()
        return img.toFloat().resample('bilinear').reproject(proj, None, 10).clip(geom)

    return ee.Image(ee.Algorithms.If(count.gt(0), _build(), _empty()))
