import pytest


ee = pytest.importorskip("ee")

try:  # pragma: no cover - skip if Earth Engine is unavailable
    ee.Initialize()
except Exception:  # pragma: no cover - environment guard
    pytest.skip("Earth Engine initialization failed", allow_module_level=True)


def test_compute_ndvi_single_band_mask():
    from app.services.indices import compute_ndvi

    img = ee.Image.cat(
        [
            ee.Image.constant(0.7).rename("B8"),
            ee.Image.constant(0.3).rename("B4"),
            ee.Image.constant(0.1).rename("B2"),
        ]
    )
    ndvi = compute_ndvi(img)
    assert ndvi.bandNames().getInfo() == ["NDVI"]
    assert ee.Image(ndvi.mask()).bandNames().size().getInfo() == 1
