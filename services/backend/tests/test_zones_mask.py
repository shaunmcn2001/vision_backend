from __future__ import annotations

from datetime import date

import ee
import pytest

from app import gee
from app.services import zones

_AOI_COORDS = [
    [149.0, -35.0],
    [149.001, -35.0],
    [149.001, -35.001],
    [149.0, -35.001],
    [149.0, -35.0],
]


def _geometry() -> ee.Geometry:
    return ee.Geometry.Polygon([_AOI_COORDS])


def _run_case(start: date, end: date) -> tuple[int, int, int]:
    geometry = _geometry()
    months = zones._months_from_dates(start, end)
    composites, _skipped, _meta = zones._build_composite_series(
        geometry,
        months,
        start,
        end,
        zones.DEFAULT_CLOUD_PROB_MAX,
    )
    if not composites:
        pytest.skip("No Sentinel-2 composites available for test window")

    ndvi_images = [zones._compute_ndvi(image) for _, image in composites]
    stats = zones._ndvi_temporal_stats(ndvi_images)
    cv_image = stats["cv"]

    thresholds = [zones.DEFAULT_CV_THRESHOLD]
    for fallback in zones.STABILITY_THRESHOLD_SEQUENCE:
        if fallback > zones.DEFAULT_CV_THRESHOLD + 1e-9:
            thresholds.append(fallback)

    stability_mask = zones._stability_mask(
        cv_image,
        geometry,
        thresholds,
        zones.MIN_STABILITY_SURVIVAL_RATIO,
        zones.DEFAULT_SCALE,
    )

    mean_count = zones._pixel_count(
        stats["mean"],
        geometry,
        context="zones test mean count",
        scale=zones.DEFAULT_SCALE,
    )
    mask_count = zones._pixel_count(
        stability_mask,
        geometry,
        context="zones test stability count",
        scale=zones.DEFAULT_SCALE,
    )

    ndvi_stats = {**stats, "stability": stability_mask}
    zone_image, _thresholds = zones._build_percentile_zones(
        ndvi_stats=ndvi_stats,
        geometry=geometry,
        n_classes=zones.DEFAULT_N_CLASSES,
        smooth_radius_m=zones.DEFAULT_SMOOTH_RADIUS_M,
        open_radius_m=zones.DEFAULT_OPEN_RADIUS_M,
        close_radius_m=zones.DEFAULT_CLOSE_RADIUS_M,
        min_mapping_unit_ha=0,
    )
    zone_count = zones._pixel_count(
        zone_image.updateMask(zone_image),
        geometry,
        context="zones test zone count",
        scale=zones.DEFAULT_SCALE,
    )

    return mean_count, mask_count, zone_count


@pytest.fixture(scope="module")
def ee_ready():
    try:
        gee.initialize()
    except RuntimeError:
        pytest.skip("Earth Engine credentials not configured")
    zones.set_apply_stability(True)
    return True


def test_stability_mask_counts_short_range(ee_ready):
    mean_count, mask_count, zone_count = _run_case(
        start=date(2023, 1, 1),
        end=date(2023, 3, 31),
    )

    assert mean_count > 0
    assert mask_count > 0
    assert zone_count > 0


def test_stability_mask_counts_long_range(ee_ready):
    mean_count, mask_count, zone_count = _run_case(
        start=date(2022, 1, 1),
        end=date(2023, 12, 31),
    )

    assert mean_count > 0
    assert mask_count > 0
    assert zone_count > 0
