import ee
import pytest
from shapely.geometry import Polygon, mapping

from app.services import zones_workflow
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


@pytest.mark.skipif(not _EE_AVAILABLE, reason=_SKIP_REASON or "Earth Engine unavailable")
def test_mean_masks_union_of_partial_months(monkeypatch):
    geom = _geom()

    def _monthly_partial_masks(**kwargs):
        base = ee.Image.pixelLonLat()
        mask_left = base.select("longitude").lt(153.005)
        mask_right = base.select("longitude").gte(153.005)

        img_left = (
            ee.Image.constant([0.2, 0.6])
            .rename(["B4", "B8"])
            .updateMask(mask_left)
        )
        img_right = (
            ee.Image.constant([0.2, 0.6])
            .rename(["B4", "B8"])
            .updateMask(mask_right)
        )

        return ee.ImageCollection([img_left, img_right])

    monkeypatch.setattr(
        zones_workflow.gee, "monthly_sentinel2_collection", _monthly_partial_masks
    )

    mean_img = _build_mean_ndvi_for_zones(geom, "2021-06-01", "2021-08-31")

    left_stats = mean_img.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=ee.Geometry.Point(153.0025, -27.5055).buffer(30),
        scale=10,
        maxPixels=1e9,
        bestEffort=True,
    ).getInfo()
    right_stats = mean_img.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=ee.Geometry.Point(153.0075, -27.5055).buffer(30),
        scale=10,
        maxPixels=1e9,
        bestEffort=True,
    ).getInfo()

    assert left_stats.get("NDVI_mean") is not None
    assert right_stats.get("NDVI_mean") is not None


@pytest.mark.skipif(not _EE_AVAILABLE, reason=_SKIP_REASON or "Earth Engine unavailable")
def test_long_term_mean_sets_metadata():
    geom = _geom()
    mean_img = zones_workflow.long_term_mean_ndvi(geom, "2021-06-01", "2021-08-31")

    bands = mean_img.bandNames().getInfo()
    assert bands == ["NDVI"]

    props = mean_img.toDictionary().getInfo()
    assert props.get("period_start") == "2021-06-01"
    assert props.get("period_end") == "2021-08-31"
    assert "ndvi_stdDev" in props
    assert "ndvi_low_variation" in props

    nominal_scale = mean_img.projection().nominalScale().getInfo()
    assert nominal_scale == pytest.approx(10, rel=0.1)


@pytest.mark.skipif(not _EE_AVAILABLE, reason=_SKIP_REASON or "Earth Engine unavailable")
def test_long_term_mean_raises_on_empty_collection(monkeypatch):
    geom = _geom()

    def _empty_monthly_collection(**kwargs):
        return ee.ImageCollection([])

    monkeypatch.setattr(zones_workflow.gee, "monthly_sentinel2_collection", _empty_monthly_collection)

    with pytest.raises(ValueError):
        zones_workflow.long_term_mean_ndvi(geom, "2021-01-01", "2021-01-31")
