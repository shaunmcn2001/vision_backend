from __future__ import annotations
import ee
from .ee_utils import ensure_list


def stability_mask_from_cv(
    cv_image: ee.Image,
    region: ee.Geometry,
    thresholds,  # scalar or list
    scale=10,
    tile_scale=4,
    min_survival_ratio=0.0,
) -> ee.Image:
    """
    total = pixel count of cv over region
    for t in thresholds:
        m = (cv <= t)
        keep m only if surviving/total >= min_survival_ratio
    combined = max over kept masks
    if total == 0 -> pass-through (1) so tiny AOIs don't get nuked
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

    def _one(t):
        t = ee.Number(t)
        m = cv_image.lte(t)  # Image
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
        # IMPORTANT: both branches return an Image
        return ee.Image(ee.Algorithms.If(ratio.gte(min_survival_ratio), m, ee.Image(0)))

    masks = tlist.map(_one)  # list of Images only
    combined = ee.ImageCollection.fromImages(masks).max()
    pass_through = ee.Image(1)
    return ee.Image(ee.Algorithms.If(total.lte(0), pass_through, combined)).selfMask()
