import ee
import pytest
from shapely.geometry import Polygon, mapping

from app.services.zones import _build_mean_ndvi_for_zones, _classify_smooth_and_polygonize

try:
    ee.Initialize(project=None)
    _EE_AVAILABLE = True
    _SKIP_REASON = ""
except Exception as exc:  # pragma: no cover - depends on environment credentials
    _EE_AVAILABLE = False
    _SKIP_REASON = f"Earth Engine init failed: {exc}"


def _geom() -> ee.Geometry:
    poly = Polygon([
        (153.0, -27.5),
        (153.01, -27.5),
        (153.01, -27.51),
        (153.0, -27.51),
        (153.0, -27.5),
    ])
    return ee.Geometry(mapping(poly))


@pytest.mark.skipif(not _EE_AVAILABLE, reason=_SKIP_REASON or "Earth Engine unavailable")
def test_mean_and_classes():
    geom = _geom()
    mean_img = _build_mean_ndvi_for_zones(geom, "2021-06-01", "2021-08-31")
    stats = mean_img.reduceRegion(
        ee.Reducer.minMax(),
        geom,
        10,
        maxPixels=1e9,
        bestEffort=True,
    ).getInfo()
    assert "NDVI_mean_min" in stats and "NDVI_mean_max" in stats

    classified, vectors = _classify_smooth_and_polygonize(
        mean_img,
        geom,
        n_zones=5,
        mmu_ha=1.0,
        smooth_radius_px=1,
    )
    assert classified.bandNames().getInfo() == ["zone"]
    assert vectors.size().getInfo() >= 0
