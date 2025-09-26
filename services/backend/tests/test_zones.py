from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.api.zones import ProductionZonesRequest, create_production_zones
from app.services import zones


def _sample_polygon() -> dict:
    return {
        "type": "Polygon",
        "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]],
    }


def test_export_prefix_formats_months_and_name():
    prefix = zones.export_prefix("Field A", ["2024-03", "2024-04", "2024-04"])
    assert prefix == "zones/PROD_202403_202404_Field_A_zones"


def test_resolve_export_bucket_prefers_env(monkeypatch):
    monkeypatch.delenv("GEE_GCS_BUCKET", raising=False)
    monkeypatch.setenv("GCS_BUCKET", "primary-bucket")
    assert zones.resolve_export_bucket() == "primary-bucket"

    monkeypatch.delenv("GCS_BUCKET", raising=False)
    with pytest.raises(RuntimeError):
        zones.resolve_export_bucket()


def test_production_zones_request_normalises_months():
    request = ProductionZonesRequest(
        aoi_geojson=_sample_polygon(),
        aoi_name="  Demo Field  ",
        months=["2024-03", "2024-03", "2024-05"],
    )
    assert request.months == ["2024-03", "2024-05"]
    assert request.aoi_name == "Demo Field"


def test_create_production_zones_endpoint(monkeypatch):
    monkeypatch.setattr(zones, "build_zone_artifacts", lambda *args, **kwargs: SimpleNamespace())

    def _fake_exports(_artifacts, **kwargs):
        return {
            "raster": SimpleNamespace(id="task_r"),
            "vectors": SimpleNamespace(id="task_v"),
            "stats": SimpleNamespace(id="task_s"),
        }

    monkeypatch.setattr(zones, "start_zone_exports", _fake_exports)
    monkeypatch.setattr(zones, "resolve_export_bucket", lambda explicit=None: "zones-bucket")
    monkeypatch.setattr(zones, "export_prefix", lambda *args, **kwargs: "zones/PROD_202403_202405_demo_zones")
    monkeypatch.setattr(zones, "month_bounds", lambda months: (months[0], months[-1]))

    request = ProductionZonesRequest(
        aoi_geojson=_sample_polygon(),
        aoi_name="Demo",
        months=["2024-03", "2024-05"],
        indices_for_zoning=["NDVI", "NDRE"],
    )

    response = create_production_zones(request)
    assert response["bucket"] == "zones-bucket"
    assert response["paths"]["raster"].endswith("demo_zones.tif")
    assert response["tasks"]["raster"]["id"] == "task_r"

