from __future__ import annotations

import os

import ee

EPS = 1e-6
MAX_SAMPLE = max(int(os.getenv("MAX_SAMPLE_PIXELS", "4000")), 1)
EE_TILE_SCALE = int(os.getenv("EE_TILE_SCALE", "4"))
EE_MAXPIXELS = float(os.getenv("EE_MAXPIXELS", "10000000000000"))


def _s2_mask_scl_only(img: ee.Image) -> ee.Image:
    scl = img.select("SCL")
    ok = (
        scl.neq(3)
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )
    return img.updateMask(ok)


def _compute_ndvi(img: ee.Image) -> ee.Image:
    return img.normalizedDifference(["B8", "B4"]).rename("NDVI").toFloat()


def _fetch_s2_sr(aoi: ee.Geometry, start: str, end: str) -> ee.ImageCollection:
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start, end)
    )


def _ndvi_collection_scene(
    aoi: ee.Geometry, start: str, end: str, mask_fn=_s2_mask_scl_only
) -> ee.ImageCollection:
    collection = _fetch_s2_sr(aoi, start, end).map(mask_fn)
    ndvi_col = collection.map(_compute_ndvi)

    def keep_if_has_data(im: ee.Image) -> ee.Image:
        cnt = (
            im.mask()
            .reduceRegion(
                ee.Reducer.sum(), aoi, 10, maxPixels=MAX_SAMPLE, bestEffort=True
            )
            .values()
            .reduce(ee.Reducer.sum())
        )
        return ee.Image(ee.Algorithms.If(cnt.gt(0), im, None))

    return ee.ImageCollection(ndvi_col.map(keep_if_has_data))


def build_mean_ndvi(aoi: ee.Geometry, start: str, end: str):
    ndvi_col = _ndvi_collection_scene(aoi, start, end)
    valid = ndvi_col.count().gt(0)
    mean = ndvi_col.mean().updateMask(valid).clip(aoi).rename("NDVI_mean").toFloat()
    return mean, ndvi_col.size()


def thresholds_memory_safe(ndvi_mean: ee.Image, aoi: ee.Geometry, num_zones: int = 5) -> ee.List:
    pct_list = [int(100 * i / num_zones) for i in range(1, num_zones)]
    pct = ndvi_mean.reduceRegion(
        reducer=ee.Reducer.percentile(
            pct_list, maxBuckets=2048, minBucketWidth=1e-4, maxRaw=1e6
        ),
        geometry=aoi,
        scale=10,
        bestEffort=True,
        maxPixels=MAX_SAMPLE,
        tileScale=EE_TILE_SCALE,
    )
    keys = ee.List(pct_list).map(
        lambda p: ee.String("NDVI_mean").cat("_p").cat(ee.Number(p).format())
    )
    values = keys.map(lambda key: ee.Number(ee.Dictionary(pct).get(key)))
    ok = values.indexOf(None).eq(-1)

    def _strict_inc(vs: ee.List) -> ee.List:
        vs = ee.List(vs)

        def step(idx, acc):
            acc = ee.List(acc)
            cur = ee.Number(vs.get(idx))
            prev = ee.Number(
                ee.Algorithms.If(acc.size().gt(0), acc.get(-1), cur.subtract(10))
            )
            next_val = ee.Number(
                ee.Algorithms.If(cur.lte(prev), prev.add(EPS), cur)
            )
            return acc.add(next_val)

        return ee.List(
            ee.List.sequence(0, vs.size().subtract(1)).iterate(step, ee.List([]))
        )

    def _fallback():
        mm = ndvi_mean.reduceRegion(
            ee.Reducer.minMax(),
            aoi,
            10,
            bestEffort=True,
            maxPixels=MAX_SAMPLE,
            tileScale=EE_TILE_SCALE,
        )
        vmin = ee.Number(mm.get("NDVI_mean_min"))
        vmax = ee.Number(mm.get("NDVI_mean_max"))
        vmin = ee.Number(ee.Algorithms.If(vmin, vmin, -0.1))
        vmax = ee.Number(ee.Algorithms.If(vmax, vmax, 0.7))
        same = vmin.eq(vmax)
        vmin2 = ee.Number(ee.Algorithms.If(same, vmin.subtract(0.1), vmin)).max(-1)
        vmax2 = ee.Number(ee.Algorithms.If(same, vmax.add(0.1), vmax)).min(1)
        step = vmax2.subtract(vmin2).divide(num_zones)
        return ee.List.sequence(1, num_zones - 1).map(
            lambda k: vmin2.add(step.multiply(k))
        )

    return ee.List(ee.Algorithms.If(ok, _strict_inc(values), _fallback()))


def classify_from_thresholds(ndvi_mean: ee.Image, thrs: ee.List) -> ee.Image:
    thresholds = ee.List(thrs)
    base = ndvi_mean.rename("NDVI_mean")

    def assign(idx, img):
        idx = ee.Number(idx)
        prev_idx = idx.subtract(1)
        prev_threshold = ee.Number(
            ee.Algorithms.If(prev_idx.gte(0), thresholds.get(prev_idx), None)
        )
        current_threshold = ee.Number(thresholds.get(idx))
        class_value = idx.add(1)
        img = ee.Image(img)
        mask = ee.Algorithms.If(
            prev_threshold,
            base.gte(prev_threshold).And(base.lt(current_threshold)),
            base.lt(current_threshold),
        )
        return img.where(ee.Image(mask), class_value)

    initial = ee.Image(0).rename("zones").updateMask(base.mask())
    zones = ee.Image(
        ee.List.sequence(0, thresholds.size().subtract(1)).iterate(assign, initial)
    )
    last_idx = thresholds.size().subtract(1)
    last_threshold = ee.Number(thresholds.get(last_idx))
    zones = zones.where(base.gte(last_threshold), thresholds.size().add(1))
    return zones.toInt()


__all__ = [
    "build_mean_ndvi",
    "classify_from_thresholds",
    "thresholds_memory_safe",
]
