from __future__ import annotations

import pytest

from fastapi import HTTPException

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
        months=["2024-05", "2024-03", "2024-03"],
    )
    assert request.months == ["2024-03", "2024-05"]
    assert request.aoi_name == "Demo Field"


def test_create_production_zones_endpoint(monkeypatch):
    def _fake_export(_aoi, _name, months, **kwargs):
        return {
            "paths": {
                "raster": "zones/PROD_202403_202405_demo_zones.tif",
                "vectors": "zones/PROD_202403_202405_demo_zones.shp",
                "vector_components": {
                    "shp": "zones/PROD_202403_202405_demo_zones.shp",
                    "dbf": "zones/PROD_202403_202405_demo_zones.dbf",
                    "shx": "zones/PROD_202403_202405_demo_zones.shx",
                    "prj": "zones/PROD_202403_202405_demo_zones.prj",
                },
                "zonal_stats": "zones/PROD_202403_202405_demo_zones_zonal_stats.csv",
            },
            "tasks": {
                "raster": {"id": "task_r", "state": "READY", "destination_uri": "gs://zones/demo.tif"},
                "vectors": {"id": "task_v", "state": "READY", "destination_uri": "gs://zones/demo.shp"},
                "zonal_stats": {"id": "task_s", "state": "READY", "destination_uri": "gs://zones/demo.csv"},
            },
            "metadata": {"used_months": months, "skipped_months": [], "mmu_applied": True},
            "prefix": "zones/PROD_202403_202405_demo_zones",
            "bucket": "zones-bucket",
        }

    monkeypatch.setattr(zones, "export_selected_period_zones", _fake_export)

    request = ProductionZonesRequest(
        aoi_geojson=_sample_polygon(),
        aoi_name="Demo",
        months=["2024-03", "2024-05"],
        n_classes=5,
    )

    response = create_production_zones(request)
    assert response["ok"] is True
    assert response["ym_start"] == "2024-03"
    assert response["ym_end"] == "2024-05"
    assert response["paths"]["raster"].endswith("demo_zones.tif")
    assert response["paths"]["vector_components"]["dbf"].endswith("demo_zones.dbf")
    assert response["prefix"].endswith("demo_zones")

    raster_task = response["tasks"]["raster"]
    assert raster_task["id"] == "task_r"
    assert raster_task["state"] == "READY"
    assert raster_task["destination_uri"].endswith("demo.tif")


def test_create_production_zones_requires_bucket_for_gcs(monkeypatch):
    def _fail_export(*_args, **_kwargs):  # pragma: no cover - should not run
        raise AssertionError("export_selected_period_zones should not be called")

    monkeypatch.setattr(zones, "export_selected_period_zones", _fail_export)
    monkeypatch.delenv("GEE_GCS_BUCKET", raising=False)
    monkeypatch.delenv("GCS_BUCKET", raising=False)

    request = ProductionZonesRequest(
        aoi_geojson=_sample_polygon(),
        aoi_name="Demo",
        months=["2024-03"],
        export_target="gcs",
    )

    with pytest.raises(HTTPException) as excinfo:
        create_production_zones(request)

    assert excinfo.value.status_code == 400
    assert (
        excinfo.value.detail
        == "A GCS bucket must be provided when export_target is 'gcs'."
    )

