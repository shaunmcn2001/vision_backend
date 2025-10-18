
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import ee

def _mask_s2(img: ee.Image) -> ee.Image:
    scl = img.select('SCL')
    good = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
    return img.updateMask(good)

def _add_indices(img: ee.Image) -> ee.Image:
    ndvi = img.normalizedDifference(['B8','B4']).rename('NDVI')
    ndre = img.normalizedDifference(['B8','B5']).rename('NDRE')
    osavi = img.expression('((N-R)/(N+R+0.16))*1.16', {'N': img.select('B8'), 'R': img.select('B4')}).rename('OSAVI')
    return img.addBands([ndvi, ndre, osavi])

def _s2(geom, start, end):
    return (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(geom).filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50))
            .map(_mask_s2).map(_add_indices))

def _gauss(img: ee.Image, sigma_m: int = 20) -> ee.Image:
    if sigma_m is None or sigma_m <= 0:
        return img
    k = ee.Kernel.gaussian(radius=sigma_m*3, sigma=sigma_m, units='meters', normalize=True)
    return img.convolve(k)

def _robust_z(img: ee.Image, geom, band: str) -> ee.Image:
    b = img.select(band)
    p = b.reduceRegion(ee.Reducer.percentile([25,50,75]), geom, 10, maxPixels=1e13, bestEffort=True)
    med = ee.Number(p.get(band + "_p50"))
    p25 = ee.Number(p.get(band + "_p25"))
    p75 = ee.Number(p.get(band + "_p75"))
    iqr = p75.subtract(p25)
    sd = iqr.divide(1.349)
    med_img = ee.Image.constant(ee.Number(ee.Algorithms.If(ee.Algorithms.IsEqual(med, None), 0, med)))
    sd_img = ee.Image.constant(ee.Number(ee.Algorithms.If(ee.Algorithms.IsEqual(sd, None), 0.1, ee.Algorithms.If(sd.abs().gt(1e-6), sd, 0.1))))
    return b.subtract(med_img).divide(sd_img).rename('z')

def _stage_mean(geom, start, end, band: str, blur_m: int = 20) -> ee.Image:
    mean = _s2(geom, start, end).select(band).mean().clip(geom)
    return _gauss(mean, blur_m).rename(band)

def compute_advanced_layers(aoi_geom: ee.Geometry, seasons: List[Dict[str, Any]], breaks: List[float]):
    # Build per-season composites
    season_imgs = []
    for row in seasons:
        crop = (row.get('crop') or '').lower()
        sow = row.get('sowing_date')
        har = row.get('harvest_date')
        # Safe dates
        if not sow or not har:
            continue
        sow_d = ee.Date.parse('YYYY-MM-dd', sow) if '-' in sow else ee.Date.parse('dd/MM/yyyy', sow)
        har_d = ee.Date.parse('YYYY-MM-dd', har) if '-' in har else ee.Date.parse('dd/MM/yyyy', har)
        # Windows (defaults)
        E = 40; L = 30
        earlyStart = sow_d; earlyEnd = sow_d.advance(E, 'day')
        lateEnd = har_d; lateStart = har_d.advance(-L, 'day')
        prodStart = earlyEnd; prodEnd = lateStart
        # Sanity order
        prodEnd = ee.Date(ee.Algorithms.If(prodEnd.millis().lt(prodStart.millis()), prodStart, prodEnd))
        lateStart = ee.Date(ee.Algorithms.If(lateStart.millis().lt(prodEnd.millis()), prodEnd, lateStart))
        # Means
        mE = _stage_mean(aoi_geom, earlyStart, earlyEnd, 'OSAVI')
        mP = _stage_mean(aoi_geom, prodStart,  prodEnd,  'NDVI')
        mL = _stage_mean(aoi_geom, lateStart,  lateEnd,  'NDRE')
        # z
        zE = _robust_z(mE, aoi_geom, 'OSAVI')
        zP = _robust_z(mP, aoi_geom, 'NDVI')
        zL = _robust_z(mL, aoi_geom, 'NDRE')
        # weights (simple default)
        comp = zE.multiply(ee.Image.constant(0.25)).add(zP.multiply(ee.Image.constant(0.5))).add(zL.multiply(ee.Image.constant(0.25)))
        season_imgs.append(comp.rename('c'))
    if not season_imgs:
        raise ee.EEException("No valid season rows.")
    coll = ee.ImageCollection(season_imgs)
    lt_comp = coll.median().rename('c').clip(aoi_geom)

    # classify with fixed breaks -> 5 classes
    b1, b2, b3, b4 = [ee.Number(x) for x in breaks]
    c = lt_comp.select('c')
    zones = (c.where(c.lte(b1), 1)
               .where(c.gt(b1).And(c.lte(b2)), 2)
               .where(c.gt(b2).And(c.lte(b3)), 3)
               .where(c.gt(b3).And(c.lte(b4)), 4)
               .where(c.gt(b4), 5)
               .rename('zone').toInt())

    vectors = (zones.rename('ZONE')
               .reduceToVectors(geometry=aoi_geom, scale=10, geometryType='polygon',
                                labelProperty='ZONE', maxPixels=1e13, eightConnected=True, tileScale=4))
    return lt_comp, zones, ee.FeatureCollection(vectors)
