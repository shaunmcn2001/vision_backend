from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import ee


def _prepare_collection(
    collection: ee.ImageCollection | Sequence[ee.Image] | Iterable[ee.Image],
    band_name: str,
    rename_base: str,
) -> ee.ImageCollection:
    if hasattr(collection, "map"):
        base = collection  # assume already an ImageCollection-like object
    else:
        base = ee.ImageCollection(collection)

    def _select_band(image: ee.Image) -> ee.Image:
        return ee.Image(image).select([band_name]).toFloat().rename(rename_base)

    return base.map(_select_band)


def temporal_stats(
    collection: ee.ImageCollection | Sequence[ee.Image] | Iterable[ee.Image],
    *,
    band_name: str,
    rename_prefix: str | None = None,
    mean_band_name: str | None = None,
) -> Mapping[str, ee.Image]:
    """Compute temporal statistics for ``band_name`` while preserving masks."""

    rename_base = rename_prefix or band_name
    prepared = _prepare_collection(collection, band_name, rename_base)

    raw_sum = prepared.reduce(ee.Reducer.sum())
    raw_count = prepared.reduce(ee.Reducer.count())
    raw_median = prepared.reduce(ee.Reducer.median())
    raw_std = prepared.reduce(ee.Reducer.stdDev())

    valid_mask = raw_count.gt(0)
    safe_count = raw_count.where(raw_count.eq(0), 1)

    mean_unmasked = raw_sum.divide(safe_count)
    mean_name = mean_band_name or f"{rename_base}_mean"
    mean = mean_unmasked.updateMask(valid_mask).rename(mean_name)

    median = raw_median.updateMask(valid_mask).rename(f"{rename_base}_median")
    std = raw_std.updateMask(valid_mask).rename(f"{rename_base}_stdDev")

    epsilon = ee.Image.constant(1e-6)
    mean_abs = mean_unmasked.abs()
    safe_denominator = mean_abs.where(mean_abs.lt(epsilon), epsilon)
    cv_raw = raw_std.divide(safe_denominator)
    cv = cv_raw.where(mean_unmasked.lte(0), 0).updateMask(valid_mask).rename(
        f"{rename_base}_cv"
    )

    return {
        "collection": prepared,
        "raw_sum": raw_sum,
        "raw_count": raw_count,
        "raw_median": raw_median,
        "raw_std": raw_std,
        "mean_unmasked": mean_unmasked,
        "mean": mean,
        "median": median,
        "std": std,
        "cv": cv,
        "valid_mask": valid_mask,
    }
