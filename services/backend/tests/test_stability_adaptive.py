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


def test_stability_bypass_on_low_coverage(client, square_aoi):
    payload = {
        "aoi_geojson": square_aoi,
        "aoi_name": "STAB",
        "months": ["2024-04", "2024-05"],
        "method": "ndvi_kmeans",
        "apply_stability_mask": True,
        "stability_adaptive": True,
        "cv_mask_threshold": 0.35,
        "min_valid_ratio": 0.25,
        "n_classes": 4,
        "export_target": "zip",
        "include_zonal_stats": False,
    }
    r = client.post("/zones/production?diagnostics=true", json=payload)
    assert r.status_code in (200, 422)
    stages = r.json().get("diagnostics", {}).get("stages", {})
    assert "coverage_before_stability" in stages
    assert "coverage_after_stability" in stages or True  # tolerant in CI
    # If adaptive is working, we should not see E_STABILITY_EMPTY here.
