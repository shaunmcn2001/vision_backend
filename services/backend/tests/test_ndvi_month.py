import base64
from datetime import date
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.main import app  # noqa: E402
from app.services.zones_workflow import (  # noqa: E402
    NdviMonthlyStat,
    compute_yearly_ndvi_averages,
    last_year_monthly_stats,
    monthly_stats_to_rows,
    yearly_averages_to_rows,
)


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


def test_ndvi_month_response_includes_aggregates_and_downloads(monkeypatch):
    client = TestClient(app)
    from app.api import products

    dec = FakeImage("ndvi_2023-12")
    jan = FakeImage("ndvi_2024-01")
    feb = FakeImage("ndvi_2024-02")

    monkeypatch.setattr(
        products,
        "monthly_ndvi",
        lambda *_args, **_kwargs: FakeCollection([dec, jan, feb]),
    )
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
    mean_lookup = {
        "ndvi_2023-12": 0.3,
        "ndvi_2024-01": 0.1,
        "ndvi_2024-02": None,
        "mean": 0.2,
    }
    monkeypatch.setattr(
        products,
        "_mean_ndvi_value",
        lambda image, _geometry: mean_lookup.get(getattr(image, "name", "")),
    )

    payload = {
        "aoi": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
        "start": date(2023, 12, 15).isoformat(),
        "end": date(2024, 2, 20).isoformat(),
    }
    response = client.post("/api/products/ndvi-month", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "downloads" in body
    downloads = body["downloads"]
    assert downloads["ndvi_2023-12_raw_geotiff"].endswith("ndvi_2023_12_raw_ndvi.tif")
    assert downloads["ndvi_2023-12_colour_geotiff"].endswith("ndvi_2023_12_colour_ndvi.tif")
    assert downloads["ndvi_mean_raw_geotiff"].endswith("ndvi_mean_raw.tif")
    assert downloads["ndvi_mean_colour_geotiff"].endswith("ndvi_mean_colour.tif")
    assert body["overallMeanNdvi"] == 0.2

    assert body["yearlyAverages"] == [
        {"year": 2023, "meanNdvi": 0.3},
        {"year": 2024, "meanNdvi": 0.1},
    ]
    last_year = body["lastYearMonthlyAverages"]
    assert [entry["label"] for entry in last_year] == ["ndvi_2024-01", "ndvi_2024-02"]
    assert last_year[1]["meanNdvi"] is None

    csv_downloads = body["csvDownloads"]
    expected_keys = {
        "ndvi_monthly_stats_csv",
        "ndvi_yearly_stats_csv",
        "ndvi_last_year_monthly_stats_csv",
    }
    assert expected_keys.issubset(csv_downloads.keys())
    for value in csv_downloads.values():
        assert value.startswith("data:text/csv;base64,")
    monthly_csv = csv_downloads["ndvi_monthly_stats_csv"]
    decoded = base64.b64decode(monthly_csv.split(",", 1)[1]).decode()
    assert "ndvi_2023-12" in decoded
    assert "ndvi_2024-02" in decoded


def test_ndvi_aggregation_helpers_multi_year():
    stats = [
        NdviMonthlyStat(year=2023, month=12, label="ndvi_2023-12", mean_ndvi=0.3),
        NdviMonthlyStat(year=2024, month=1, label="ndvi_2024-01", mean_ndvi=0.1),
        NdviMonthlyStat(year=2024, month=2, label="ndvi_2024-02", mean_ndvi=None),
    ]

    yearly = compute_yearly_ndvi_averages(stats)
    assert [(item.year, item.mean_ndvi) for item in yearly] == [(2023, 0.3), (2024, 0.1)]

    last_year = last_year_monthly_stats(stats)
    assert [item.month for item in last_year] == [1, 2]

    monthly_rows = monthly_stats_to_rows(stats)
    assert monthly_rows[0]["label"] == "ndvi_2023-12"

    yearly_rows = yearly_averages_to_rows(yearly)
    assert yearly_rows[1]["mean_ndvi"] == 0.1
