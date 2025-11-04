from datetime import date
from types import SimpleNamespace
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.main import app  # noqa: E402


def test_advanced_zones_downloads(monkeypatch):
    client = TestClient(app)
    from app.api import products

    class FakeFeatureCollection:
        def getInfo(self):
            return {"type": "FeatureCollection", "features": []}

    fake_result = SimpleNamespace(
        composite="composite-image",
        zones_raster="zones-image",
        raw_zones=FakeFeatureCollection(),
        dissolved_zones=FakeFeatureCollection(),
    )

    monkeypatch.setattr(products, "ensure_ee", lambda: None)
    monkeypatch.setattr(products, "to_geometry", lambda aoi: aoi)
    monkeypatch.setattr(products, "compute_advanced_layers", lambda *_args, **_kwargs: fake_result)
    monkeypatch.setattr(products, "_attach_stats", lambda *_args, **_kwargs: "stats-fc")
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
    monkeypatch.setattr(products, "table_csv_url", lambda *_args, **_kwargs: "https://example.com/stats.csv")

    payload = {
        "aoi": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
        "breaks": [-1.0, -0.3, 0.3, 1.0],
        "seasons": [
            {
                "sowingDate": date(2023, 5, 1).isoformat(),
                "harvestDate": date(2023, 11, 1).isoformat(),
            }
        ],
    }

    response = client.post("/api/zones/advanced", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["preview"]["zones"]["tile"]["token"] == "tok"
    assert body["downloads"]["zonesGeotiff"].endswith("zones.tif")
    assert len(body["downloads"]) == 5
    assert body["vectorsGeojson"]["type"] == "FeatureCollection"
    assert body["classCount"] == 5
