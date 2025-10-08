from __future__ import annotations

import pytest

from app.api import zones as zones_api
from app.api.zones import ProductionZonesRequest
from app.services import zones as zone_service


@pytest.fixture
def square_aoi():
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


def test_kmeans_records_input_health(monkeypatch, square_aoi):
    def fake_export_selected_period_zones(*args, diagnostics=False, **kwargs):
        base = {
            "paths": {},
            "tasks": {},
            "metadata": {"used_months": ["2025-07"], "skipped_months": []},
            "artifacts": object(),
            "working_dir": "/tmp/zones",
        }
        if diagnostics:
            base["diagnostics"] = {"stages": {"ndvi_input_health": {"band_ok": True}}}
        return base

    monkeypatch.setattr(
        zone_service, "export_selected_period_zones", fake_export_selected_period_zones
    )

    request = ProductionZonesRequest(
        aoi_geojson=square_aoi,
        aoi_name="KMEANS_HEALTH",
        method="ndvi_kmeans",
        months=["2025-07", "2025-08"],
        n_classes=4,
        export_target="zip",
        include_zonal_stats=False,
    )

    response = zones_api.create_production_zones(request, diagnostics=True)
    diag = response.get("diagnostics", {}).get("stages", {})
    assert "ndvi_input_health" in diag
