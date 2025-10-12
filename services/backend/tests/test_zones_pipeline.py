import ee
import pytest
from shapely.geometry import Polygon, mapping

from app.services.zones import (
    _build_mean_ndvi_for_zones,
    _classify_smooth_and_polygonize,
)


try:
    ee.Initialize(project=None)
except Exception:
    pytestmark = pytest.mark.skip(reason="Earth Engine credentials required for smoke test")


def _geom():
    return ee.Geometry(
        mapping(
            Polygon(
                [
                    (153.0, -27.5),
                    (153.01, -27.5),
                    (153.01, -27.51),
                    (153.0, -27.51),
                    (153.0, -27.5),
                ]
            )
        )
    )


def test_zones_pipeline_smoke():
    g = _geom()
    mean = _build_mean_ndvi_for_zones(g, "2021-06-01", "2021-08-31")
    stats = mean.reduceRegion(
        ee.Reducer.minMax(), g, 10, bestEffort=True, maxPixels=1e9
    ).getInfo()
    assert "NDVI_mean_min" in stats
    cls, vec = _classify_smooth_and_polygonize(mean, g)
    assert cls.bandNames().getInfo() == ["zone"]
