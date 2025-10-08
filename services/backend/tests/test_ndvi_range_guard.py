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


@pytest.mark.parametrize(
    "months",
    [["2025-07", "2025-08"]],
)
def test_ndvi_range_guard_present(client, square_aoi, months):
    payload = {
        "aoi_geojson": square_aoi,
        "aoi_name": "RANGE_GUARD",
        "months": months,
        "method": "ndvi_kmeans",
        "n_classes": 4,
        "export_target": "zip",
        "include_zonal_stats": False,
    }
    response = client.post("/zones/production?diagnostics=true", json=payload)
    assert response.status_code in (200, 422)
    diagnostics = response.json().get("diagnostics", {}).get("stages", {})
    assert "ndvi_input_health" in diagnostics
