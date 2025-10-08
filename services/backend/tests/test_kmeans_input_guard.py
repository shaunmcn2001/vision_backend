import pytest

pytest.importorskip("httpx", reason="httpx is required for FastAPI TestClient")

from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def square_aoi() -> dict:
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [-0.0005, -0.0005],
                [-0.0005, 0.0005],
                [0.0005, 0.0005],
                [0.0005, -0.0005],
                [-0.0005, -0.0005],
            ]
        ],
    }


def test_kmeans_records_input_health(client: TestClient, square_aoi: dict) -> None:
    payload = {
        "aoi_geojson": square_aoi,
        "aoi_name": "KMEANS_HEALTH",
        "months": ["2025-07", "2025-08"],
        "method": "ndvi_kmeans",
        "n_classes": 4,
        "export_target": "zip",
        "include_zonal_stats": False,
    }
    r = client.post("/zones/production?diagnostics=true", json=payload)
    assert r.status_code in (200, 422)
    diag = r.json().get("diagnostics", {}).get("stages", {})
    assert "ndvi_input_health" in diag
