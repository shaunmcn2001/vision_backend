from __future__ import annotations

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


def test_per_month_diag_attached_when_debug(
    client: TestClient, square_aoi: dict
) -> None:
    payload = {
        "aoi_geojson": square_aoi,
        "aoi_name": "DEBUG_DIAG",
        "method": "ndvi_kmeans",
        "months": ["2025-07", "2025-08"],
        "n_classes": 4,
        "export_target": "zip",
        "include_zonal_stats": False,
        "debug_dump": True,
    }
    response = client.post("/zones/production?diagnostics=true", json=payload)
    assert response.status_code in (200, 422)
    stages = response.json().get("diagnostics", {}).get("stages", {})
    assert "ndvi_per_month" in stages


def test_kmeans_ndvi_input_health_present(client: TestClient, square_aoi: dict) -> None:
    payload = {
        "aoi_geojson": square_aoi,
        "aoi_name": "KMEANS_HEALTH",
        "method": "ndvi_kmeans",
        "months": ["2025-07", "2025-08"],
        "n_classes": 4,
        "export_target": "zip",
        "include_zonal_stats": False,
    }
    response = client.post("/zones/production?diagnostics=true", json=payload)
    assert response.status_code in (200, 422)
    stages = response.json().get("diagnostics", {}).get("stages", {})
    assert "ndvi_input_health" in stages
