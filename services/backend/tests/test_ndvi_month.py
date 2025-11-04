from datetime import date
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.main import app  # noqa: E402


class FakeNumber:
    def __init__(self, value: int):
        self._value = value

    def getInfo(self) -> int:
        return self._value


class FakeList:
    def __init__(self, items):
        self._items = items

    def get(self, index: int):
        return self._items[index]


class FakeCollection:
    def __init__(self, images):
        self._images = images

    def size(self):
        return FakeNumber(len(self._images))

    def toList(self, _size: int):
        return FakeList(self._images)


class FakeImage:
    def __init__(self, name: str):
        self.name = name

    def visualize(self, **_kwargs):
        return FakeImage(f"{self.name}-vis")


def test_ndvi_month_includes_downloads(monkeypatch):
    client = TestClient(app)
    from app.api import products

    jan = FakeImage("jan")
    feb = FakeImage("feb")
    monkeypatch.setattr(products, "monthly_ndvi", lambda *_args, **_kwargs: FakeCollection([jan, feb]))
    monkeypatch.setattr(products, "mean_ndvi", lambda *_args, **_kwargs: FakeImage("mean"))
    monkeypatch.setattr(
        products,
        "create_tile_session",
        lambda image, **_kwargs: {
            "token": f"token-{getattr(image, 'name', 'image')}",
            "url_template": "/api/tiles/test/{z}/{x}/{y}",
            "min_zoom": 0,
            "max_zoom": 14,
        },
    )
    monkeypatch.setattr(
        products,
        "image_geotiff_url",
        lambda _image, _aoi, *, name, **_kwargs: f"https://example.com/{name}.tif",
    )
    monkeypatch.setattr(products.ee, "Image", lambda value: value)

    payload = {
        "aoi": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
        "start": date(2024, 1, 1).isoformat(),
        "end": date(2024, 2, 28).isoformat(),
    }
    response = client.post("/api/products/ndvi-month", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "downloads" in body
    downloads = body["downloads"]
    assert downloads["ndvi_2024-01_raw_geotiff"].endswith("ndvi_2024_01_raw_ndvi.tif")
    assert downloads["ndvi_2024-01_colour_geotiff"].endswith("ndvi_2024_01_colour_ndvi.tif")
    assert downloads["ndvi_mean_raw_geotiff"].endswith("ndvi_mean_raw.tif")
    assert downloads["ndvi_mean_colour_geotiff"].endswith("ndvi_mean_colour.tif")
