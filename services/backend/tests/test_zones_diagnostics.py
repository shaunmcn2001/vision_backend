import pytest

from app.api import zones as zones_api
from app.api.zones import ProductionZonesRequest


@pytest.fixture
def sample_aoi():
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [149.0, -35.0],
                [149.0005, -35.0],
                [149.0005, -35.0005],
                [149.0, -35.0005],
                [149.0, -35.0],
            ]
        ],
    }


def test_diagnostics_toggle(monkeypatch, sample_aoi):
    captured = {}

    def fake_export(*args, diagnostics=False, **kwargs):
        captured["diagnostics"] = diagnostics
        base_payload = {
            "paths": {},
            "tasks": {},
            "metadata": {"used_months": ["2025-07"], "skipped_months": []},
        }
        if diagnostics:
            base_payload["diagnostics"] = {"stages": {"inputs": {"area_m2": 123}}}
        return base_payload

    monkeypatch.setattr(
        zones_api.zone_service,
        "export_selected_period_zones",
        fake_export,
    )

    request = ProductionZonesRequest(
        aoi_geojson=sample_aoi,
        aoi_name="TEST",
        method="ndvi_percentiles",
        months=["2025-07", "2025-08"],
        cloud_prob_max=60,
        n_classes=5,
        cv_mask_threshold=0.30,
        mmu_ha=1.0,
        smooth_radius_m=20,
        open_radius_m=20,
        close_radius_m=30,
        simplify_tol_m=0.0,
        simplify_buffer_m=0.0,
        export_target="zip",
        include_zonal_stats=True,
    )

    response = zones_api.create_production_zones(request, diagnostics=True)
    assert captured.get("diagnostics") is True
    assert "diagnostics" in response
    assert "stages" in response["diagnostics"]
