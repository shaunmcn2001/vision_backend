import ee
import pytest
from shapely.geometry import Polygon
from shapely.geometry import mapping as shp_mapping

from app.services.zones import _build_mean_ndvi_for_zones

try:
    ee.Initialize(project=None)
    _EE_INITIALIZED = True
    _EE_SKIP_REASON = ""
except Exception as exc:
    _EE_INITIALIZED = False
    _EE_SKIP_REASON = f"Earth Engine init failed: {exc}"


def _geom():
    poly = Polygon([
        (153.0, -27.5),
        (153.01, -27.5),
        (153.01, -27.51),
        (153.0, -27.51),
        (153.0, -27.5),
    ])
    return ee.Geometry(shp_mapping(poly))


@pytest.mark.skipif(not _EE_INITIALIZED, reason=_EE_SKIP_REASON or "Earth Engine unavailable")
def test_mean_ndvi_exists_and_has_range():
    geom = _geom()
    img = _build_mean_ndvi_for_zones(geom, "2021-06-01", "2021-08-31")
    stats = img.reduceRegion(
        reducer=ee.Reducer.minMax(),
        geometry=geom,
        scale=10,
        maxPixels=1e9,
        bestEffort=True,
    ).getInfo()
    assert "NDVI_mean_min" in stats and "NDVI_mean_max" in stats
