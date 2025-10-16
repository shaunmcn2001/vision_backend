import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.api import routes_ndvi as routes


def _setup_fake_environment(monkeypatch, downloads):
    class FakeGeometry:
        def __init__(self, payload):
            self.payload = payload

        def toGeoJSONString(self) -> str:
            return json.dumps(self.payload)

    class FakeFormatted:
        def __init__(self, text: str):
            self.text = text

        def getInfo(self) -> str:
            return self.text

    class FakeNumber:
        def __init__(self, value):
            self.value = value

        def format(self, fmt: str | None = None) -> FakeFormatted:
            text = format(self.value, fmt) if fmt else str(self.value)
            return FakeFormatted(text)

        def getInfo(self):
            return self.value

    class FakeImage:
        def __init__(self, name: str, props: dict | None = None):
            self.name = name
            self.props = props or {}
            self.reproject_history: list[tuple[str, float]] = []
            self.reproject_called = False

        def get(self, key):
            return self.props.get(key)

        def getDownloadURL(self, params):
            downloads.append(
                {
                    "name": self.name,
                    "params": params,
                    "reproject": list(self.reproject_history),
                    "reproject_called": self.reproject_called,
                    "type": type(self).__name__,
                }
            )
            return f"https://example.com/{self.name}"

        def toInt(self):
            return self

        def toFloat(self):
            return self

        def resample(self, _method):
            return self

        def reproject(self, crs, _transform, scale):
            self.reproject_history.append((crs, scale))
            self.reproject_called = True
            return self

    class FakeList:
        def __init__(self, items):
            self._items = items

        def get(self, idx):
            return self._items[idx]

    class FakeImageCollection:
        def __init__(self, images):
            self._images = images

        def toList(self, _size):
            return FakeList(self._images)

        def size(self):
            return FakeNumber(len(self._images))

    fake_data = ModuleType("ee.data")

    def _download_id(_vectors, params):
        return f"id-{params['fileFormat']}"

    def _download_url(download_id):
        return f"https://example.com/{download_id}"

    fake_data.getTableDownloadId = _download_id
    fake_data.getTableDownloadUrl = _download_url

    fake_ee = ModuleType("ee")
    fake_ee.Geometry = lambda payload: FakeGeometry(payload)
    fake_ee.Image = (
        lambda value: value
        if isinstance(value, FakeImage)
        else FakeImage(getattr(value, "name", str(value)), getattr(value, "props", None))
    )
    fake_ee.Number = lambda value: FakeNumber(value)
    fake_ee.data = fake_data

    monthly_images = [FakeImage("monthly-1", {"year": 2024, "month": 1})]
    monthly_collection = FakeImageCollection(monthly_images)

    monkeypatch.setitem(sys.modules, "ee", fake_ee)
    monkeypatch.setattr(routes, "ee", fake_ee)
    monkeypatch.setattr(routes.zw, "init_ee", lambda: None)
    monkeypatch.setattr(routes.zw, "get_s2_sr_collection", lambda *a, **kw: "collection")
    monkeypatch.setattr(routes.zw, "monthly_ndvi_mean", lambda *a, **kw: monthly_collection)
    monkeypatch.setattr(routes.zw, "long_term_mean_ndvi", lambda *a, **kw: FakeImage("mean"))
    monkeypatch.setattr(
        routes.zw,
        "classify_zones",
        lambda *a, **kw: {"classified": FakeImage("classified"), "breaks": [0.2, 0.8]},
    )
    monkeypatch.setattr(routes.zw, "vectorize_zones", lambda *a, **kw: "vectors")


def _build_request(**overrides) -> routes.NDVIRequest:
    payload = {
        "aoi": routes.AOI(
            type="Polygon",
            coordinates=[[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]],
        ),
        "start": "2024-01-01",
        "end": "2024-01-31",
    }
    payload.update(overrides)
    return routes.NDVIRequest(**payload)


def test_generate_ndvi_defaults_to_native_crs(monkeypatch):
    downloads: list[dict] = []
    _setup_fake_environment(monkeypatch, downloads)

    request = _build_request()
    response = routes.generate_ndvi(request)

    assert response["mean_ndvi"]["url"].startswith("https://example.com/")
    assert downloads, "expected downloads to be recorded"
    for entry in downloads:
        assert "crs" not in entry["params"], entry
        assert not entry["reproject_called"], entry


def test_generate_ndvi_respects_export_crs_override(monkeypatch):
    downloads: list[dict] = []
    _setup_fake_environment(monkeypatch, downloads)

    request = _build_request(export_crs="EPSG:32632")
    routes.generate_ndvi(request)

    assert {entry["params"]["crs"] for entry in downloads} == {"EPSG:32632"}


def test_generate_ndvi_reprojects_before_download(monkeypatch):
    downloads: list[dict] = []
    _setup_fake_environment(monkeypatch, downloads)

    request = _build_request(export_crs="EPSG:32632", export_scale=20)
    routes.generate_ndvi(request)

    assert downloads, "expected downloads to be recorded"
    for record in downloads:
        assert any(
            crs == "EPSG:32632" and scale == pytest.approx(20.0)
            for crs, scale in record["reproject"]
        ), record


def test_generate_ndvi_sets_finite_nodata(monkeypatch):
    downloads: list[dict] = []
    _setup_fake_environment(monkeypatch, downloads)

    request = _build_request()
    routes.generate_ndvi(request)

    assert downloads, "expected downloads to be recorded"
    for record in downloads:
        options = record["params"].get("formatOptions")
        assert options and "noData" in options, record
        nodata = options["noData"]
        assert nodata is not None and nodata == pytest.approx(-9999.0)
