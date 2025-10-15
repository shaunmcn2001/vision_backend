"""
zones_workflow.py
-----------------
Drop-in NDVI workflows for your backend:

Outputs it enables:
  1) Raster NDVI for **each month** in a selected date range
  2) Raster NDVI **mean** over the whole selected period
  3) **Classified** NDVI zones raster (default 5 quantile zones)
  4) **Vector** polygons of the zones

Design goals:
- Light cloud mask (s2cloudless + minimal SCL masking)
- Work at native S2 10 m during processing; export CRS default EPSG:4326
- Safe reducers (bestEffort + tileScale)
- Self-contained (depends on `earthengine-api` only)
"""

from __future__ import annotations

import ee
from typing import Dict, Any, List, Optional

# -----------------------------
# EE Init
# -----------------------------

def init_ee(project: Optional[str] = None) -> None:
    if project:
        ee.Initialize(project=project)
    else:
        ee.Initialize()

# -----------------------------
# Helpers
# -----------------------------

def to_ndvi(img: ee.Image) -> ee.Image:
    """Compute NDVI for Sentinel-2 SR."""
    return img.normalizedDifference(['B8', 'B4']).rename('NDVI')


def _s2cloudless(aoi: ee.Geometry, start: str, end: str) -> ee.ImageCollection:
    return (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
            .filterBounds(aoi)
            .filterDate(start, end))


def _s2_sr(aoi: ee.Geometry, start: str, end: str) -> ee.ImageCollection:
    return (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(aoi)
            .filterDate(start, end)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80)))


def _light_cloud_mask(s2: ee.Image, cloudprob: ee.Image,
                      cloud_prob_max: int = 40,
                      mask_scl_shadow_snow: bool = True) -> ee.Image:
    prob = cloudprob.select('probability').lte(cloud_prob_max)
    scl = s2.select('SCL')
    if mask_scl_shadow_snow:
        not_shadow = scl.neq(3)   # Shadow
        not_snow   = scl.neq(11)  # Snow
        mask = prob.And(not_shadow).And(not_snow)
    else:
        mask = prob
    qa60 = s2.select('QA60')
    not_saturated = qa60.bitwiseAnd(1 << 1).eq(0)
    not_nodata    = qa60.bitwiseAnd(1 << 0).eq(0)
    return s2.updateMask(mask.And(not_saturated).And(not_nodata))


def get_s2_sr_collection(aoi: ee.Geometry, start: str, end: str,
                         cloud_prob_max: int = 40,
                         mask_scl_shadow_snow: bool = True) -> ee.ImageCollection:
    s2 = _s2_sr(aoi, start, end)
    s2c = _s2cloudless(aoi, start, end)

    # Join by closest time (maxDifference) to ensure a cloudprob match
    time_join = ee.Filter.maxDifference(
        difference=1000 * 60 * 60,  # 1 hour
        leftField='system:time_start',
        rightField='system:time_start'
    )
    save = ee.Join.saveBest('cloudprob', time_join)
    joined = ee.ImageCollection(save.apply(s2, s2c))

    def attach_and_mask(img):
        cp = ee.Image(img.get('cloudprob'))
        return to_ndvi(_light_cloud_mask(img, cp, cloud_prob_max, mask_scl_shadow_snow))

    return joined.map(attach_and_mask)


def monthly_ndvi_mean(collection: ee.ImageCollection) -> ee.ImageCollection:
    """Monthly mean NDVI composites from an NDVI-bearing collection."""
    def with_tags(img):
        date = ee.Date(img.get('system:time_start'))
        return img.set({'year': date.get('year'), 'month': date.get('month')})

    tagged = collection.map(with_tags)
    years = ee.List(ee.Dictionary(tagged.aggregate_histogram('year')).keys().sort())
    months = ee.List.sequence(1, 12)

    def ym_to_img(y):
        y = ee.Number(y)
        def m_to_img(m):
            m = ee.Number(m)
            subset = tagged.filter(ee.Filter.And(ee.Filter.eq('year', y), ee.Filter.eq('month', m)))
            comp = subset.select('NDVI').mean().rename('NDVI')
            date = ee.Date.fromYMD(y, m, 1)
            return comp.set({'year': y, 'month': m, 'system:time_start': date.millis()})
        return ee.ImageCollection(months.map(m_to_img))

    return ee.ImageCollection(years.map(ym_to_img)).flatten()


def long_term_mean_ndvi(collection: ee.ImageCollection) -> ee.Image:
    return collection.select('NDVI').mean().rename('NDVI')


def classify_zones(ndvi_img: ee.Image,
                   aoi: ee.Geometry,
                   n_zones: int = 5,
                   method: str = 'quantile',
                   smooth_radius_m: int = 30,
                   mmu_pixels: int = 50) -> Dict[str, Any]:
    nd = ndvi_img.select('NDVI').clip(aoi)
    if method.lower() == 'quantile':
        pct = [100 * i / n_zones for i in range(1, n_zones)]  # e.g., [20,40,60,80]
        stats = nd.reduceRegion(
            reducer=ee.Reducer.percentile(pct),
            geometry=aoi, scale=10, maxPixels=1e13, bestEffort=True, tileScale=4
        )
        breaks = ee.List([stats.get(f'NDVI_p{int(p)}') for p in pct])
    else:
        mnmx = nd.reduceRegion(ee.Reducer.minMax(), aoi, 10, maxPixels=1e13, bestEffort=True, tileScale=4)
        mn = ee.Number(mnmx.get('NDVI_min'))
        mx = ee.Number(mnmx.get('NDVI_max'))
        step = mx.subtract(mn).divide(n_zones)
        breaks = ee.List([mn.add(step.multiply(i)) for i in range(1, n_zones)])

    def bin_classify(img, brks):
        b0 = ee.Number(brks.get(0))
        b1 = ee.Number(brks.get(1))
        b2 = ee.Number(brks.get(2))
        b3 = ee.Number(brks.get(3)) if n_zones >= 5 else ee.Number(1e9)
        z1 = img.lt(b0)
        z2 = img.gte(b0).And(img.lt(b1))
        z3 = img.gte(b1).And(img.lt(b2))
        z4 = img.gte(b2).And(img.lt(b3))
        z5 = img.gte(b3)
        return (z1.multiply(1)  # 1..5
                .add(z2.multiply(2))
                .add(z3.multiply(3))
                .add(z4.multiply(4))
                .add(z5.multiply(5))).rename('ZONE')

    classified = bin_classify(nd, breaks)

    # Smooth + MMU
    if smooth_radius_m and smooth_radius_m > 0:
        radius_px = ee.Number(smooth_radius_m).divide(10.0)
        kernel = ee.Kernel.circle(radius_px, 'pixels', True)
        classified = classified.focal_mode(kernel=kernel, iterations=1)

    if mmu_pixels and mmu_pixels > 0:
        # GEE limit: maxSize<=1024
        cpc = classified.connectedPixelCount(maxSize=1024, eightConnected=True)
        classified = classified.updateMask(cpc.gte(mmu_pixels))

    return {'classified': classified, 'breaks': breaks}


def vectorize_zones(classified: ee.Image,
                    aoi: ee.Geometry,
                    simplify_tolerance_m: float = 0.0) -> ee.FeatureCollection:
    zones = (classified.rename('ZONE')
             .clip(aoi)
             .reduceToVectors(geometry=aoi, scale=10, geometryType='polygon',
                              labelProperty='ZONE', maxPixels=1e13,
                              eightConnected=True, tileScale=4))
    fc = ee.FeatureCollection(zones)
    if simplify_tolerance_m and simplify_tolerance_m > 0:
        fc = fc.map(lambda f: f.setGeometry(f.geometry().simplify(simplify_tolerance_m)))
    return fc
