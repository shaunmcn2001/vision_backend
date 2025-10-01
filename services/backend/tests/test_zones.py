from __future__ import annotations

from datetime import date
from typing import Any

import pytest
from fastapi import HTTPException

from app.api.zones import ProductionZonesRequest, create_production_zones
from app.services import zones


def _sample_polygon() -> dict:
    return {
        "type": "Polygon",
        "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]],
    }


def _run_prepare_with_counts(
    monkeypatch,
    survival_counts,
    *,
    cv_threshold,
    total_pixels=100,
    composite_metadata=None,
):
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

    metadata = composite_metadata or {"composite_mode": "monthly"}

    def _fake_build_series(_geometry, months, start_date, end_date, _cloud_prob_max):
        assert start_date == date(2024, 1, 1)
        assert end_date == date(2024, 1, 31)
        return [(months[0], object())], [], metadata

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
    monkeypatch.setattr(zones, "_build_composite_series", _fake_build_series)
    monkeypatch.setattr(zones, "_compute_ndvi", _fake_compute_ndvi)
    monkeypatch.setattr(zones, "_ndvi_temporal_stats", _fake_temporal_stats)
    monkeypatch.setattr(zones, "_ndvi_cv", _fake_cv)
    monkeypatch.setattr(zones, "_stability_mask", _fake_stability_mask)
    monkeypatch.setattr(
        zones,
        "_build_percentile_zones",
        lambda **_kwargs: (_FakeZoneImage(), [0.2, 0.4, 0.6, 0.8]),
    )
    monkeypatch.setattr(zones, "_prepare_vectors", lambda *_args, **_kwargs: _FakeVectors())
    monkeypatch.setattr(zones, "area_ha", lambda *_args, **_kwargs: 10)

    artifacts, metadata = zones._prepare_selected_period_artifacts(
        _sample_polygon(),
        geometry=object(),
        months=["2024-01"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
        cloud_prob_max=20,
        n_classes=5,
        cv_mask_threshold=cv_threshold,
        min_mapping_unit_ha=0.5,
        smooth_radius_m=15,
        open_radius_m=10,
        close_radius_m=10,
        simplify_tol_m=5,
        simplify_buffer_m=0,
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
    assert request.start_date == date(2024, 3, 1)
    assert request.end_date == date(2024, 5, 31)


def test_production_zones_request_accepts_range():
    request = ProductionZonesRequest(
        aoi_geojson=_sample_polygon(),
        aoi_name="Range",
        start_month="2024-01",
        end_month="2024-03",
    )

    assert request.months == ["2024-01", "2024-02", "2024-03"]
    assert request.start_date == date(2024, 1, 1)
    assert request.end_date == date(2024, 3, 31)


def test_production_zones_request_accepts_dates():
    request = ProductionZonesRequest(
        aoi_geojson=_sample_polygon(),
        aoi_name="Dates",
        start_date="2024-02-15",
        end_date="2024-04-02",
    )

    assert request.start_date == date(2024, 2, 15)
    assert request.end_date == date(2024, 4, 2)
    assert request.months == ["2024-02", "2024-03", "2024-04"]


def test_production_zones_request_rejects_conflicting_inputs():
    with pytest.raises(ValueError) as excinfo:
        ProductionZonesRequest(
            aoi_geojson=_sample_polygon(),
            aoi_name="Conflict",
            months=["2024-01"],
            start_month="2024-01",
        )

    assert "Provide only one of months[]" in str(excinfo.value)


def test_production_zones_request_rejects_inverted_dates():
    with pytest.raises(ValueError) as excinfo:
        ProductionZonesRequest(
            aoi_geojson=_sample_polygon(),
            aoi_name="Bad",
            start_date="2024-05-01",
            end_date="2024-04-01",
        )

    assert "end_date must be on or after start_date" in str(excinfo.value)


def test_create_production_zones_endpoint(monkeypatch):
    def _fake_export(_aoi, _name, months, *, start_date=None, end_date=None, **kwargs):
        assert start_date == date(2024, 3, 1)
        assert end_date == date(2024, 5, 31)
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
                "palette": list(zones.ZONE_PALETTE[:5]),
                "percentile_thresholds": [0.1, 0.2, 0.3, 0.4],
                "stability": {
                    "initial_threshold": 0.25,
                    "final_threshold": 1.0,
                    "thresholds_tested": [0.25, 0.5, 1.0],
                    "survival_ratio": 0.32,
                    "surviving_pixels": 32,
                    "total_pixels": 100,
                    "target_ratio": zones.MIN_STABILITY_SURVIVAL_RATIO,
                    "low_confidence": False,
                },
                "debug": {
                    "stability": {
                        "survival_ratios": [0.1, 0.22, 0.32],
                        "thresholds_tested": [0.25, 0.5, 1.0],
                    }
                },
            },
            "prefix": "zones/PROD_202403_202405_demo_zones",
            "bucket": "zones-bucket",
            "palette": list(zones.ZONE_PALETTE[:5]),
            "thresholds": [0.1, 0.2, 0.3, 0.4],
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
    assert debug["retry_thresholds"] == [0.25, 0.5, 1.0]
    stability = debug["stability"]
    assert stability["initial_threshold"] == 0.25
    assert stability["final_threshold"] == 1.0
    assert stability["survival_ratio"] == 0.32
    assert stability["survival_ratios"] == [0.1, 0.22, 0.32]
    assert response["palette"] == list(zones.ZONE_PALETTE[:5])
    assert response["thresholds"] == [0.1, 0.2, 0.3, 0.4]


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

    def _fake_export(_aoi, _name, months, *, start_date=None, end_date=None, **_kwargs):
        captured["months"] = months
        captured["start_date"] = start_date
        captured["end_date"] = end_date
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
    assert captured["start_date"] == date(2024, 1, 1)
    assert captured["end_date"] == date(2024, 2, 29)
    assert response["debug"]["requested_months"] == ["2024-01", "2024-02"]
    assert response["debug"]["stability"] is None


def test_create_production_zones_with_date_request(monkeypatch):
    captured = {}

    def _fake_export(_aoi, _name, months, *, start_date=None, end_date=None, **_kwargs):
        captured["months"] = months
        captured["start_date"] = start_date
        captured["end_date"] = end_date
        return {
            "paths": {},
            "tasks": {},
            "metadata": {"used_months": months, "skipped_months": []},
            "prefix": "zones/demo",
        }

    monkeypatch.setattr(zones, "export_selected_period_zones", _fake_export)

    request = ProductionZonesRequest(
        aoi_geojson=_sample_polygon(),
        aoi_name="Date Demo",
        start_date="2024-03-10",
        end_date="2024-04-05",
    )

    response = create_production_zones(request)

    assert captured["months"] == ["2024-03", "2024-04"]
    assert captured["start_date"] == date(2024, 3, 10)
    assert captured["end_date"] == date(2024, 4, 5)
    assert response["debug"]["requested_months"] == ["2024-03", "2024-04"]
    assert response["debug"]["stability"] is None


def test_prepare_selected_period_artifacts_reports_stability(monkeypatch):
    survival_counts = {0.25: 5, 0.5: 10, 1.0: 18, 1.5: 28, 2.0: 35}
    artifacts, metadata = _run_prepare_with_counts(
        monkeypatch, survival_counts, cv_threshold=0.25
    )

    assert artifacts.zone_image.__class__.__name__ == "_FakeZoneImage"
    assert metadata["percentile_thresholds"] == [0.2, 0.4, 0.6, 0.8]
    assert metadata["palette"] == list(zones.ZONE_PALETTE[:5])
    assert metadata["composite_mode"] == "monthly"

    stability = metadata["stability"]
    assert stability["thresholds_tested"] == [0.25]
    assert stability["final_threshold"] == pytest.approx(0.25)
    assert stability["survival_ratio"] == pytest.approx(0.05)
    assert stability["low_confidence"] is False
    assert metadata["low_confidence"] is False

    debug = metadata["debug"]["stability"]
    assert debug["thresholds_tested"] == [0.25]
    assert debug["low_confidence"] is False
    assert debug["survival_ratios"] == pytest.approx([0.05])


def test_prepare_selected_period_artifacts_retains_initial_threshold(monkeypatch):
    survival_counts = {0.5: 30, 1.0: 60, 1.5: 80, 2.0: 90}
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


def test_prepare_selected_period_artifacts_records_scene_mode(monkeypatch):
    survival_counts = {0.25: 25, 0.5: 40, 1.0: 60}
    _artifacts, metadata = _run_prepare_with_counts(
        monkeypatch,
        survival_counts,
        cv_threshold=0.25,
        composite_metadata={"composite_mode": "scene", "scene_count": 7},
    )

    assert metadata["composite_mode"] == "scene"
    assert metadata["scene_count"] == 7
    assert metadata["palette"] == list(zones.ZONE_PALETTE[:5])
    assert metadata["percentile_thresholds"] == [0.2, 0.4, 0.6, 0.8]
    assert metadata["low_confidence"] is False


def test_apply_cleanup_uses_meter_operations(monkeypatch):
    calls: list[tuple[str, tuple[Any, ...]]] = []

    class _FakeMask:
        def __init__(self):
            self.calls: list[tuple[str, tuple[Any, ...]]] = []

        def focal_max(self, radius, units, iterations):
            self.calls.append(("max", (radius, units, iterations)))
            return self

        def focal_min(self, radius, units, iterations):
            self.calls.append(("min", (radius, units, iterations)))
            return self

        def Not(self):
            self.calls.append(("not", ()))
            return "mask_not"

    class _FakeImage:
        def __init__(self):
            self.mask_obj = _FakeMask()

        def focal_mode(self, radius, units, iterations):
            calls.append(("mode", (radius, units, iterations)))
            return self

        def focal_min(self, radius, units, iterations):
            calls.append(("min", (radius, units, iterations)))
            return self

        def focal_max(self, radius, units, iterations):
            calls.append(("max", (radius, units, iterations)))
            return self

        def where(self, *_args):
            calls.append(("where", _args))
            return self

        def mask(self):
            return self.mask_obj

        def updateMask(self, *_args):
            calls.append(("updateMask", _args))
            return self

        def clip(self, geom):
            calls.append(("clip", (geom,)))
            return self

    class _FakeAreaImage:
        def __init__(self):
            self.lt_calls: list[tuple[Any, ...]] = []

        def lt(self, value):
            self.lt_calls.append((value,))
            return "lt_mask"

    fake_area = _FakeAreaImage()

    def _fake_connected_component_area(classified, n_classes):
        assert n_classes == 5
        return fake_area

    fake_image = _FakeImage()
    monkeypatch.setattr(zones, "_connected_component_area", _fake_connected_component_area)

    result = zones._apply_cleanup(
        fake_image,
        geometry="geom",
        n_classes=5,
        smooth_radius_m=15,
        open_radius_m=10,
        close_radius_m=10,
        min_mapping_unit_ha=1.0,
    )

    assert result is fake_image
    meter_calls = [entry for entry in calls if entry[0] in {"mode", "min", "max"}]
    assert all(args[1] == "meters" for _, args in meter_calls)
    radii = [args[0] for _, args in meter_calls]
    assert 15 in radii
    assert radii.count(10) >= 2
    assert fake_image.mask_obj.calls[0][1][1] == "meters"
    assert fake_area.lt_calls[0][0] == pytest.approx(10_000.0)


def test_simplify_vectors_sets_area_and_buffer(monkeypatch):
    class _FakeNumber:
        def __init__(self, value):
            self.value = value

        def divide(self, denom):
            return self.value / denom

    class _FakeGeometry:
        def __init__(self):
            self.simplify_args: list[float] = []
            self.buffer_args: list[float] = []

        def simplify(self, maxError):
            self.simplify_args.append(maxError)
            return self

        def buffer(self, distance):
            self.buffer_args.append(distance)
            return self

        def area(self, maxError):
            assert maxError == 1
            return _FakeNumber(12_000)

    class _FakeFeature:
        def __init__(self, geom):
            self.geom = geom
            self.props = {"zone": 3}

        def geometry(self):
            return self.geom

        def setGeometry(self, geom):
            self.geom = geom
            return self

        def set(self, values):
            self.props.update(values)
            return self

        def get(self, key):
            return self.props[key]

    class _FakeCollection:
        def __init__(self, features):
            self.features = features

        def map(self, func):
            return _FakeCollection([func(feature) for feature in self.features])

    geometry = _FakeGeometry()
    collection = _FakeCollection([_FakeFeature(geometry)])

    class _FakeNumberWrapper:
        def __init__(self, value):
            self.value = value

        def toInt(self):
            return int(self.value)

    class _FakeEE:
        @staticmethod
        def Number(value):
            return _FakeNumberWrapper(value)

        @staticmethod
        def FeatureCollection(features):
            return features

    monkeypatch.setattr(zones, "ee", _FakeEE())

    simplified = zones._simplify_vectors(collection, tolerance_m=15, buffer_m=2)
    feature = simplified.features[0]
    assert geometry.simplify_args == [15]
    assert geometry.buffer_args == [2]
    assert feature.props["zone_id"].__class__.__name__ != "NoneType"
    assert feature.props["area_m2"].value == 12_000
    assert feature.props["area_ha"] == pytest.approx(1.2)

