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


def test_kmeans_input_health_is_recorded(client, square_aoi):
    payload = {
        "aoi_geojson": square_aoi,
        "aoi_name": "HEALTH",
        "months": ["2024-04", "2024-05"],
        "method": "ndvi_kmeans",
        "n_classes": 4,
        "export_target": "zip",
        "include_zonal_stats": False,
    }
    r = client.post("/zones/production?diagnostics=true", json=payload)
    assert r.status_code in (200, 422)
    diag = r.json().get("diagnostics", {}).get("stages", {})
    assert "ndvi_input_health" in diag


def test_no_percentiles_in_kmeans_path(client, square_aoi, monkeypatch):
    from app.services import zones as Z

    def _boom(*args, **kwargs):
        raise RuntimeError("percentiles called in kmeans path")

    monkeypatch.setattr(Z, "robust_quantile_breaks", _boom, raising=True)
    payload = {
        "aoi_geojson": square_aoi,
        "aoi_name": "NO_PERC",
        "months": ["2024-04", "2024-05"],
        "method": "ndvi_kmeans",
        "n_classes": 4,
        "export_target": "zip",
        "include_zonal_stats": False,
    }
    r = client.post("/zones/production?diagnostics=true", json=payload)
    assert r.status_code in (200, 422)


def test_coverage_diagnostic_or_error(client, square_aoi):
    payload = {
        "aoi_geojson": square_aoi,
        "aoi_name": "COVERAGE",
        "months": ["2024-04", "2024-05"],
        "method": "ndvi_kmeans",
        "n_classes": 4,
        "export_target": "zip",
        "include_zonal_stats": False,
    }
    response = client.post("/zones/production?diagnostics=true", json=payload)
    if response.status_code == 422:
        detail = response.json()
        if "detail" in detail:
            detail = detail["detail"]
        assert detail.get("code") == "E_COVERAGE_LOW"
    else:
        assert response.status_code == 200
        stages = response.json().get("diagnostics", {}).get("stages", {})
        assert "coverage_before_stability" in stages
