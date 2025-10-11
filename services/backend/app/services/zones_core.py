from __future__ import annotations

import ee

from .ee_utils import ensure_list


def normalize_ndvi_band(img):
    """Map 'NDVI_mean' or 'NDVI' onto a single 'NDVI' band."""

    names = img.bandNames()
    has_ndvi = names.contains("NDVI")
    has_mean = names.contains("NDVI_mean")
    out = ee.Image(
        ee.Algorithms.If(
            has_ndvi,
            img.select(["NDVI"]).rename(["NDVI"]),
            ee.Algorithms.If(
                has_mean,
                img.select(["NDVI_mean"]).rename(["NDVI"]),
                img,
            ),
        )
    )
    return ee.Image(out).select(["NDVI"]).rename(["NDVI"])


def stats_stack(ndvi_ic):
    """Return mean/stdDev/cv stack for NDVI collections."""

    mean = ndvi_ic.mean().rename("NDVI_mean")
    stdv = ndvi_ic.reduce(ee.Reducer.stdDev()).rename("NDVI_stdDev")
    safe_mean = ee.Image(mean).where(ee.Image(mean).abs().lt(1e-6), 1e-6)
    cv = ee.Image(stdv).divide(safe_mean).rename("NDVI_cv")
    return mean.addBands(stdv).addBands(cv)


def stability_mask_from_cv(
    cv_image,
    region,
    thresholds,
    scale=10,
    tile_scale=4,
    min_survival_ratio=0.0,
):
    """
    Build stability mask mirroring JS implementation.
    """

    total = ee.Number(
        cv_image.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=region,
            scale=scale,
            bestEffort=True,
            maxPixels=1e9,
            tileScale=tile_scale,
        )
        .values()
        .get(0)
    )

    tlist = ensure_list(thresholds)
    masks = tlist.map(
        lambda t: _stability_mask_one(
            cv_image,
            region,
            scale,
            tile_scale,
            min_survival_ratio,
            ee.Number(t),
            total,
        )
    )

    combined = ee.ImageCollection.fromImages(masks).max()
    pass_through = ee.Image(1)
    return ee.Image(ee.Algorithms.If(total.lte(0), pass_through, combined)).selfMask()


def _stability_mask_one(
    cv_image,
    region,
    scale,
    tile_scale,
    min_survival_ratio,
    t,
    total,
):
    m = cv_image.lte(t)
    surviving = ee.Number(
        m.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=region,
            scale=scale,
            bestEffort=True,
            maxPixels=1e9,
            tileScale=tile_scale,
        )
        .values()
        .get(0)
    )
    ratio = surviving.divide(total.max(1))
    return ee.Image(ee.Algorithms.If(ratio.gte(min_survival_ratio), m, ee.Image(0)))


def coverage_ratio(img, region, scale=10, tile_scale=4):
    """Compute ratio of valid pixels to theoretical total."""

    valid_sum = img.mask().reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=scale,
        bestEffort=True,
        maxPixels=1e9,
        tileScale=tile_scale,
    )
    total_px = region.area(1).divide(ee.Number(scale).pow(2))
    px_sum = ee.Number(valid_sum.values().reduce(ee.Reducer.sum()))
    return px_sum.divide(total_px)
