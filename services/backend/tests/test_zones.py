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


def _run_prepare_with_counts(monkeypatch, survival_counts, *, cv_threshold, total_pixels=100):
    class _FakeZoneImage:
        def updateMask(self, *_args, **_kwargs):
            return self

        def neq(self, _value):
            return self

        def toInt16(self):
            return self

        def rename(self, _name):
            return self

        def reproject(self, *_args, **_kwargs):
            return self

    class _FakeVectors:
        pass

    class _FakeCVImage:
        def __init__(self, total: int):
            self._total = total

        def reduceRegion(self, **_kwargs):
            return {"NDVI_cv": self._total}

    class _FakeMask:
        def __init__(self, threshold: float, count: int):
            self.threshold = threshold
            self._count = count

        def updateMask(self, *_args, **_kwargs):
            return self

        def reduceRegion(self, **_kwargs):
            return {"NDVI_cv": self._count}

    class _FakeEE:
        class Reducer:
            @staticmethod
            def count():
                return object()

    def _fake_build_monthly_composites(_geometry, months, _cloud_prob_max):
        return [(months[0], object())], []

    def _fake_compute_ndvi(_image):
        return object()

    def _fake_temporal_stats(_images):
        return {"mean": object(), "median": object(), "std": object()}

    def _fake_cv(_mean, _std):
        return _FakeCVImage(total_pixels)

    def _fake_stability_mask(_image, threshold):
        key = round(float(threshold), 3)
        if key not in survival_counts:
            raise AssertionError(f"Unexpected threshold {threshold}")
        return _FakeMask(key, survival_counts[key])

    monkeypatch.setattr(zones, "ee", _FakeEE())
    monkeypatch.setattr(zones, "_build_monthly_composites", _fake_build_monthly_composites)
    monkeypatch.setattr(zones, "_compute_ndvi", _fake_compute_ndvi)
    monkeypatch.setattr(zones, "_ndvi_temporal_stats", _fake_temporal_stats)
    monkeypatch.setattr(zones, "_ndvi_cv", _fake_cv)
    monkeypatch.setattr(zones, "_stability_mask", _fake_stability_mask)
    monkeypatch.setattr(zones, "_build_percentile_zones", lambda **_kwargs: _FakeZoneImage())
    monkeypatch.setattr(zones, "_prepare_vectors", lambda *_args, **_kwargs: _FakeVectors())
    monkeypatch.setattr(zones, "area_ha", lambda *_args, **_kwargs: 10)

    artifacts, metadata = zones._prepare_selected_period_artifacts(
        _sample_polygon(),
        geometry=object(),
        months=["2024-01"],
        cloud_prob_max=20,
        n_classes=5,
        cv_mask_threshold=cv_threshold,
        min_mapping_unit_ha=0.5,
        smooth_kernel_px=1,
        simplify_tol_m=5,
        method="ndvi_percentiles",
        sample_size=100,
        include_stats=False,
    )

    return artifacts, metadata


def test_percentile_thresholds_raise_when_mask_removes_all_pixels(monkeypatch):
    class _FakeList(list):
        def map(self, func):
            return _FakeList([func(item) for item in self])

    class _FakeKeys:
        def getInfo(self):
            return []

    class _FakeDictionary:
        def keys(self):
            return _FakeKeys()

        def get(self, _name):  # pragma: no cover - not used when raising
            return None

    class _FakeImage:
        def reduceRegion(self, **_kwargs):
            return _FakeDictionary()

    class _FakeEE:
        class Reducer:
            @staticmethod
            def percentile(percentiles, names):  # pragma: no cover - passthrough stub
                return {"percentiles": percentiles, "names": names}

        @staticmethod
        def List(values):
            return _FakeList(list(values))

        @staticmethod
        def Number(value):  # pragma: no cover - not used when raising
            return value

    fake_ee = _FakeEE()
    monkeypatch.setattr(zones, "ee", fake_ee)

    with pytest.raises(ValueError) as excinfo:
        zones._percentile_thresholds(_FakeImage(), geometry=object(), n_classes=5)

    assert str(excinfo.value) == zones.STABILITY_MASK_EMPTY_ERROR


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


def test_production_zones_request_accepts_range():
    request = ProductionZonesRequest(
        aoi_geojson=_sample_polygon(),
        aoi_name="Range",
        start_month="2024-01",
        end_month="2024-03",
    )

    assert request.months == ["2024-01", "2024-02", "2024-03"]


def test_production_zones_request_rejects_conflicting_inputs():
    with pytest.raises(ValueError) as excinfo:
        ProductionZonesRequest(
            aoi_geojson=_sample_polygon(),
            aoi_name="Conflict",
            months=["2024-01"],
            start_month="2024-01",
        )

    assert "either months[] or start_month/end_month" in str(excinfo.value)


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
            "metadata": {
                "used_months": months,
                "skipped_months": ["2024-04"],
                "mmu_applied": True,
                "stability": {
                    "initial_threshold": 0.25,
                    "final_threshold": 0.6,
                    "thresholds_tested": [0.25, 0.5, 0.6],
                    "survival_ratio": 0.32,
                    "surviving_pixels": 32,
                    "total_pixels": 100,
                    "target_ratio": zones.MIN_STABILITY_SURVIVAL_RATIO,
                    "low_confidence": False,
                },
                "debug": {
                    "stability": {
                        "survival_ratios": [0.1, 0.2, 0.32],
                        "thresholds_tested": [0.25, 0.5, 0.6],
                    }
                },
            },
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

    debug = response["debug"]
    assert debug["requested_months"] == ["2024-03", "2024-05"]
    assert debug["skipped_months"] == ["2024-04"]
    assert debug["retry_thresholds"] == [0.25, 0.5, 0.6]
    stability = debug["stability"]
    assert stability["initial_threshold"] == 0.25
    assert stability["final_threshold"] == 0.6
    assert stability["survival_ratio"] == 0.32
    assert stability["survival_ratios"] == [0.1, 0.2, 0.32]


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


def test_create_production_zones_with_range_request(monkeypatch):
    captured = {}

    def _fake_export(_aoi, _name, months, **_kwargs):
        captured["months"] = months
        return {
            "paths": {},
            "tasks": {},
            "metadata": {"used_months": months, "skipped_months": []},
            "prefix": "zones/demo",
        }

    monkeypatch.setattr(zones, "export_selected_period_zones", _fake_export)

    request = ProductionZonesRequest(
        aoi_geojson=_sample_polygon(),
        aoi_name="Demo",
        start_month="2024-01",
        end_month="2024-02",
    )

    response = create_production_zones(request)

    assert captured["months"] == ["2024-01", "2024-02"]
    assert response["debug"]["requested_months"] == ["2024-01", "2024-02"]
    assert response["debug"]["stability"] is None


def test_prepare_selected_period_artifacts_retries_thresholds(monkeypatch):
    survival_counts = {0.25: 5, 0.5: 10, 0.6: 18, 0.8: 28, 1.0: 35}
    artifacts, metadata = _run_prepare_with_counts(
        monkeypatch, survival_counts, cv_threshold=0.25
    )

    assert artifacts.zone_image.__class__.__name__ == "_FakeZoneImage"

    stability = metadata["stability"]
    assert stability["thresholds_tested"] == [0.25, 0.5, 0.6, 0.8]
    assert stability["final_threshold"] == pytest.approx(0.8)
    assert stability["survival_ratio"] == pytest.approx(0.28)
    assert stability["low_confidence"] is True
    assert metadata["low_confidence"] is True

    debug = metadata["debug"]["stability"]
    assert debug["thresholds_tested"] == [0.25, 0.5, 0.6, 0.8]
    assert debug["low_confidence"] is True
    assert debug["survival_ratios"] == pytest.approx([0.05, 0.1, 0.18, 0.28])


def test_prepare_selected_period_artifacts_retains_initial_threshold(monkeypatch):
    survival_counts = {0.5: 30, 0.6: 60, 0.8: 80, 1.0: 90}
    _artifacts, metadata = _run_prepare_with_counts(
        monkeypatch, survival_counts, cv_threshold=0.5
    )

    stability = metadata["stability"]
    assert stability["thresholds_tested"] == [0.5]
    assert stability["final_threshold"] == pytest.approx(0.5)
    assert stability["survival_ratio"] == pytest.approx(0.3)
    assert stability["low_confidence"] is False
    assert metadata["low_confidence"] is False

    debug = metadata["debug"]["stability"]
    assert debug["thresholds_tested"] == [0.5]
    assert debug["survival_ratios"] == pytest.approx([0.3])

