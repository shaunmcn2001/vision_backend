from datetime import date
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.main import app  # noqa: E402


def test_basic_zones_downloads(monkeypatch):
    client = TestClient(app)
    from app.api import products

    monkeypatch.setattr(products, "ensure_ee", lambda: None)
    monkeypatch.setattr(products, "to_geometry", lambda aoi: aoi)
    monkeypatch.setattr(products, "mean_ndvi", lambda *_args, **_kwargs: "mean-image")
    monkeypatch.setattr(products, "classify_zones", lambda *_args, **_kwargs: "zones-image")
    class FakeFeatureCollection:
        def getInfo(self):
            return {"type": "FeatureCollection", "features": []}

    monkeypatch.setattr(products, "vectorize_zones", lambda *_args, **_kwargs: FakeFeatureCollection())
    monkeypatch.setattr(products, "_class_stats_fc", lambda *_args, **_kwargs: "stats-fc")
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
    monkeypatch.setattr(products, "image_geotiff_url", lambda *_args, **_kwargs: "https://example.com/zones.tif")
    monkeypatch.setattr(products, "table_shp_url", lambda *_args, **_kwargs: "https://example.com/zones.zip")
    monkeypatch.setattr(products, "table_csv_url", lambda *_args, **_kwargs: "https://example.com/zones.csv")

    payload = {
        "aoi": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
        "start": date(2024, 5, 1).isoformat(),
        "end": date(2024, 6, 1).isoformat(),
        "nClasses": 5,
    }
    response = client.post("/api/zones/basic", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["preview"]["tile"]["token"] == "tok"
    assert body["downloads"]["rasterGeotiff"].endswith("zones.tif")
    assert body["downloads"]["vectorShp"].endswith("zones.zip")
    assert body["downloads"]["statsCsv"].endswith("zones.csv")
    assert body["vectorsGeojson"]["type"] == "FeatureCollection"
    assert body["classCount"] == 5
