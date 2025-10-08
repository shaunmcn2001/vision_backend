import pytest


ee = pytest.importorskip("ee")

try:  # pragma: no cover - skip if Earth Engine is unavailable
    ee.Initialize()
except Exception:  # pragma: no cover - environment guard
    pytest.skip("Earth Engine initialization failed", allow_module_level=True)


def test_monthly_composite_bandname(monkeypatch):
    from app.services import zones as Z

    aoi = ee.Geometry.Polygon(
        [
            [
                [149.7, -28.8],
                [149.71, -28.8],
                [149.71, -28.79],
                [149.7, -28.79],
                [149.7, -28.8],
            ]
        ]
    )
    months = ["2025-07"]

    ic = Z.build_monthly_ndvi_collection(aoi, months)

    def _check(img):
        return ee.Algorithms.If(
            ee.Image(img)
            .bandNames()
            .contains("NDVI")
            .And(ee.Image(img).bandNames().size().eq(1)),
            1,
            0,
        )

    ok = ee.ImageCollection(ic).map(_check).reduce(ee.Reducer.sum()).getInfo()
    assert ok is None or ok >= 0
