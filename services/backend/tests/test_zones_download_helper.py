from __future__ import annotations

import io
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

TEST_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TEST_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services import zones
from . import fake_ee


class _FakeGeometry:
    def geometry(self):
        return self

    def getInfo(self):
        return {
            "type": "Polygon",
            "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]],
        }


class _FakeResponse:
    def __init__(self, data: bytes):
        self._buffer = io.BytesIO(data)
        self.headers = {"Content-Type": "image/tiff"}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self._buffer.close()

    def read(self, size: int | None = -1):
        return self._buffer.read(size if size is not None else -1)


def test_download_helper_preserves_native_crs_and_nodata(monkeypatch, tmp_path):
    image_context: dict[str, object] = {}
    image = fake_ee.FakeMeanImage(0.5, image_context)
    image.toFloat = lambda: image  # type: ignore[attr-defined]

    native_crs = "EPSG:4326"

    class _FakeProjection:
        def __init__(self, crs: str):
            self._crs = crs

        def crs(self):
            return SimpleNamespace(getInfo=lambda: self._crs)

        def getInfo(self):
            return {"crs": self._crs}

    image.projection = lambda: _FakeProjection(native_crs)  # type: ignore[attr-defined]

    captured_download: dict[str, object] = {}

    def _fake_get_download(params):
        captured_download.update(params)
        return "https://example.com/fake"

    image.getDownloadURL = _fake_get_download  # type: ignore[attr-defined]

    class _FakeTask:
        def start(self):
            return None

    monkeypatch.setattr(
        zones.ee,
        "batch",
        SimpleNamespace(Export=SimpleNamespace(image=SimpleNamespace(toDrive=lambda **_: _FakeTask()))),
    )
    monkeypatch.setattr(
        zones.ee,
        "Geometry",
        type("_DummyGeometry", (), {"Polygon": staticmethod(lambda coords: coords)}),
    )

    source_path = tmp_path / "source.tif"
    with rasterio.open(
        source_path,
        "w",
        driver="GTiff",
        height=1,
        width=1,
        count=1,
        dtype="float32",
        crs=native_crs,
        transform=from_origin(0, 1, 1, 1),
        nodata=-9999.0,
    ) as dataset:
        dataset.write(np.array([[0.5]], dtype=np.float32), 1)
    geotiff_bytes = source_path.read_bytes()

    monkeypatch.setattr(zones, "urlopen", lambda *args, **kwargs: _FakeResponse(geotiff_bytes))
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: _FakeResponse(geotiff_bytes))

    output_path = tmp_path / "download.tif"
    result = zones._download_image_to_path(image, _FakeGeometry(), output_path)

    assert result.path == output_path
    assert "crs" not in captured_download

    with rasterio.open(output_path) as src:
        assert src.crs is not None
        crs_text = src.crs.to_string() if hasattr(src.crs, "to_string") else str(src.crs)
        assert crs_text == native_crs
        assert src.nodata is not None
        assert np.isfinite(src.nodata)


def test_download_helper_rewrites_nonfinite_nodata(monkeypatch, tmp_path):
    image_context: dict[str, object] = {}
    image = fake_ee.FakeMeanImage(0.5, image_context)
    image.toFloat = lambda: image  # type: ignore[attr-defined]

    native_crs = "EPSG:4326"

    class _FakeProjection:
        def __init__(self, crs: str):
            self._crs = crs

        def crs(self):
            return SimpleNamespace(getInfo=lambda: self._crs)

        def getInfo(self):
            return {"crs": self._crs}

    image.projection = lambda: _FakeProjection(native_crs)  # type: ignore[attr-defined]

    captured_download: dict[str, object] = {}

    def _fake_get_download(params):
        captured_download.update(params)
        return "https://example.com/fake"

    image.getDownloadURL = _fake_get_download  # type: ignore[attr-defined]

    class _FakeTask:
        def start(self):
            return None

    monkeypatch.setattr(
        zones.ee,
        "batch",
        SimpleNamespace(Export=SimpleNamespace(image=SimpleNamespace(toDrive=lambda **_: _FakeTask()))),
    )
    monkeypatch.setattr(
        zones.ee,
        "Geometry",
        type("_DummyGeometry", (), {"Polygon": staticmethod(lambda coords: coords)}),
    )

    source_path = tmp_path / "source_inf.tif"
    with rasterio.open(
        source_path,
        "w",
        driver="GTiff",
        height=1,
        width=1,
        count=1,
        dtype="float32",
        crs=native_crs,
        transform=from_origin(0, 1, 1, 1),
        nodata=float("-inf"),
    ) as dataset:
        dataset.write(np.array([[-np.inf]], dtype=np.float32), 1)
    geotiff_bytes = source_path.read_bytes()

    monkeypatch.setattr(zones, "urlopen", lambda *args, **kwargs: _FakeResponse(geotiff_bytes))
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: _FakeResponse(geotiff_bytes))

    output_path = tmp_path / "download_inf.tif"
    result = zones._download_image_to_path(image, _FakeGeometry(), output_path)

    assert result.path == output_path
    assert "crs" not in captured_download

    with rasterio.open(output_path) as src:
        assert src.crs is not None
        assert np.isfinite(src.nodata)
        assert src.nodata == zones.DEFAULT_NODATA_VALUE
        arr = src.read(1, masked=True)
        assert np.ma.is_masked(arr)
        assert bool(arr.mask[0, 0])
        assert arr.data[0, 0] == zones.DEFAULT_NODATA_VALUE
        raw = src.read(1, masked=False)
        assert np.isfinite(raw).all()


def test_valid_pixel_ratio_guard_raises_for_all_nodata(tmp_path):
    nodata_value = -9999.0
    raster_path = tmp_path / "all_nodata.tif"

    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=5,
        width=5,
        count=1,
        dtype="float32",
        transform=from_origin(0, 0, 10, 10),
        nodata=nodata_value,
    ) as dataset:
        dataset.write(np.full((1, 5, 5), nodata_value, dtype=np.float32), 1)

    with pytest.raises(ValueError) as exc:
        zones._ensure_minimum_valid_pixels(raster_path, label="NDVI mean")

    message = str(exc.value)
    assert "fewer than" in message
    assert "Revisit the mask" in message

