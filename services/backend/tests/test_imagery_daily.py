from datetime import date
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.main import app  # noqa: E402


class FakeNumber:
    def __init__(self, value: float):
        self._value = value

    def multiply(self, factor: float):
        return FakeNumber(self._value * factor)

    def getInfo(self) -> float:
        return self._value


class FakeCloudMask:
    def __init__(self, fraction: float):
        self.fraction = fraction

    def eq(self, _value: int):
        return self

    def Or(self, _other):
        return self

    def reduceRegion(self, **_kwargs):
        return {"SCL": self.fraction}


class FakeImage:
    def __init__(self, cloud_fraction: float):
        self.cloud_fraction = cloud_fraction

    def clip(self, _geometry):
        return self

    def select(self, band):
        if band == "SCL":
            return FakeCloudMask(self.cloud_fraction)
        return self


class FakeImageCollection:
    def __init__(self, cloud_fraction: float):
        self.cloud_fraction = cloud_fraction

    def filterBounds(self, *_args, **_kwargs):
        return self

    def filterDate(self, *_args, **_kwargs):
        return self

    def size(self):
        return FakeNumber(1)

    def mosaic(self):
        return FakeImage(self.cloud_fraction)


def test_imagery_daily_cloud_pct(monkeypatch):
    client = TestClient(app)
    from app.api import products

    monkeypatch.setattr(products, "ensure_ee", lambda: None)
    monkeypatch.setattr(products, "to_geometry", lambda aoi: aoi)
    monkeypatch.setattr(products.ee, "ImageCollection", lambda *_args, **_kwargs: FakeImageCollection(0.42))
    monkeypatch.setattr(products.ee, "Number", FakeNumber)
    monkeypatch.setattr(
        products,
        "create_tile_session",
        lambda *_args, **_kwargs: {
            "token": "tok",
            "url_template": "/api/tiles/tok/{z}/{x}/{y}",
            "min_zoom": 0,
            "max_zoom": 14,
        },
    )

    payload = {
        "aoi": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
        "start": date(2024, 5, 1).isoformat(),
        "end": date(2024, 5, 1).isoformat(),
    }

    response = client.post("/api/imagery/daily", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["days"]) == 1
    assert abs(data["days"][0]["cloudPct"] - 42.0) < 1e-6
    assert abs(data["summary"]["avgCloudPct"] - 42.0) < 1e-6
