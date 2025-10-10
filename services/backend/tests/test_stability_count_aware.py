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


def test_stability_count_aware_bypass(client, square_aoi):
    payload = {
        "aoi_geojson": square_aoi,
        "aoi_name": "STAB2",
        "months": ["2024-03", "2024-04", "2024-05"],
        "method": "ndvi_kmeans",
        "apply_stability_mask": True,
        "stability_adaptive": True,
        "stability_enforce": False,
        "cv_mask_threshold": 0.35,
        "min_obs_for_cv": 3,
        "min_valid_ratio": 0.25,
        "n_classes": 4,
        "export_target": "zip",
        "include_zonal_stats": False,
    }
    r = client.post("/zones/production?diagnostics=true", json=payload)
    assert r.status_code in (200, 422)
    d = r.json().get("diagnostics", {}).get("stages", {})
    assert "coverage_before_stability" in d
    assert "min_obs_for_cv" in d
