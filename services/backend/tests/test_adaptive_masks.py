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


def test_per_month_records_mask_tier(client, square_aoi):
    payload = {
        "aoi_geojson": square_aoi,
        "aoi_name": "ADAPT",
        "months": ["2025-07", "2025-08"],
        "method": "ndvi_kmeans",
        "mask_mode": "adaptive",
        "min_valid_ratio": 0.25,
        "apply_stability_mask": True,
        "stability_adaptive": True,
        "cv_mask_threshold": 0.35,
        "n_classes": 4,
        "export_target": "zip",
        "include_zonal_stats": False,
    }
    response = client.post("/zones/production?diagnostics=true", json=payload)
    assert response.status_code in (200, 422)
    stages = response.json().get("diagnostics", {}).get("stages", {})
    assert "ndvi_per_month" in stages
    assert "coverage_before_stability" in stages
