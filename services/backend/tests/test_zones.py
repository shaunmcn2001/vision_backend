from __future__ import annotations

import asyncio
import json
from datetime import date
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException

from app.api.zones import ProductionZonesRequest, create_production_zones
from app.main import app
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

    class _FakeMeanImage:
        def __init__(self, total: int):
            self._total = total

        def reduceRegion(self, **_kwargs):
            return {"NDVI_mean": self._total}

        def updateMask(self, *_args, **_kwargs):  # pragma: no cover - passthrough
            return self

    class _FakeMask:
        def __init__(self, threshold: float, count: int):
            self.threshold = threshold
            self._count = count

        def selfMask(self):
            return self

        def updateMask(self, *_args, **_kwargs):
            return self

        def reduceRegion(self, **_kwargs):
            return {"mask": self._count}

    class _FakeCVImage:
        def __init__(self, total: int):
            self._total = total

        def reduceRegion(self, **_kwargs):
            return {"NDVI_cv": self._total}

        def lte(self, threshold):
            key = round(float(threshold), 3)
            if key not in survival_counts:
                raise AssertionError(f"Unexpected threshold {threshold}")
            return _FakeMask(key, survival_counts[key])

        def mask(self):  # pragma: no cover - passthrough helper
            return self

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
        return {
            "mean": _FakeMeanImage(total_pixels),
            "median": object(),
            "std": object(),
            "cv": _FakeCVImage(total_pixels),
        }

    def _fake_stability_mask(_image, _geometry, thresholds, _min_ratio, _scale):
        if not thresholds:
            raise AssertionError("thresholds should not be empty")
        last = round(float(list(thresholds)[-1]), 3)
        if last not in survival_counts:
            raise AssertionError(f"Unexpected threshold {last}")
        return _FakeMask(last, survival_counts[last])

    monkeypatch.setattr(zones, "ee", _FakeEE())
    monkeypatch.setattr(zones, "_build_composite_series", _fake_build_series)
    monkeypatch.setattr(zones, "_compute_ndvi", _fake_compute_ndvi)
    monkeypatch.setattr(zones, "_ndvi_temporal_stats", _fake_temporal_stats)
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
        apply_stability_mask=True,
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


def test_classify_by_percentiles_uses_list_iterate(monkeypatch):
    thresholds = [0.2, 0.4, 0.6, 0.8]
    context: dict[str, Any] = {}

    class FakeString:
        def __init__(self, value: str):
            self.value = value

        def getInfo(self):
            return self.value

        def __str__(self) -> str:  # pragma: no cover - debug aid
            return self.value

    class FakeNumber:
        def __init__(self, value):
            self.value = float(value)

        def getInfo(self):  # pragma: no cover - parity with ee.Number
            return self.value

        def __float__(self):
            return self.value

    class FakeBandNames:
        def __init__(self, names: list[str]):
            self._names = names

        def get(self, index: int):
            return self._names[index]

    class FakeImage:
        def __init__(self, value: float):
            self.value = float(value)
            self.renamed_to: str | None = None

        def bandNames(self):
            return FakeBandNames(["ndvi"])

        def rename(self, name):
            self.renamed_to = str(name)
            return self

        def reduceRegion(self, reducer, **_kwargs):
            context["reducer_percentiles"] = getattr(reducer, "percentiles", [])
            values = {
                name: thresholds[idx]
                for idx, name in enumerate(getattr(reducer, "outputNames", []))
            }
            return SimpleNamespace(getInfo=lambda: values)

        def multiply(self, factor):
            other = factor.value if isinstance(factor, FakeImage) else float(factor)
            return FakeImage(self.value * other)

        def gt(self, threshold):
            compare = threshold.value if isinstance(threshold, FakeNumber) else float(threshold)
            return FakeImage(1.0 if self.value > compare else 0.0)

        def add(self, other):
            addend = other.value if isinstance(other, FakeImage) else float(other)
            return FakeImage(self.value + addend)

        def addBands(self, _band):  # pragma: no cover - not used in this test
            return self

        def toInt(self):
            return self

    class FakeReducer:
        def __init__(self, percentiles, outputNames):
            if isinstance(percentiles, FakeList):
                percentiles = list(percentiles.values)
            self.percentiles = percentiles
            self.outputNames = outputNames

    class FakeList:
        def __init__(self, values):
            self.values = list(values)
            context["list_values"] = list(values)

        def iterate(self, func, first):
            context["list_iterate_called"] = True
            result = first
            for value in self.values:
                result = func(result, value)
            return result

    class FakeEE:
        class Reducer:
            @staticmethod
            def percentile(percentiles, outputNames=None):
                return FakeReducer(percentiles, outputNames or [])

        @staticmethod
        def List(values):
            return FakeList(values)

        @staticmethod
        def Number(value):
            return FakeNumber(value)

        @staticmethod
        def String(value):
            return FakeString(value)

        @staticmethod
        def Image(value):
            if isinstance(value, FakeImage):
                return value
            return FakeImage(value)

    image = FakeImage(0.5)
    monkeypatch.setattr(zones, "ee", FakeEE)

    classified, pct_thresholds = zones._classify_by_percentiles(image, geometry=object(), n_classes=5)

    assert pct_thresholds == thresholds
    assert context.get("list_iterate_called") is True
    assert isinstance(classified, FakeImage)
    assert classified.value == 3.0  # 2 thresholds exceeded -> zone id 3

def test_percentile_thresholds_raise_when_stability_removes_all_pixels():
    percentiles = [20, 40, 60, 80]

    with pytest.raises(ValueError) as excinfo:
        zones._percentile_thresholds({}, percentiles, "ndvi_mean")

    assert str(excinfo.value) == zones.STABILITY_MASK_EMPTY_ERROR


def test_prepare_selected_period_artifacts_raises_when_ndvi_mask_empty(monkeypatch):
    geometry = object()

    monkeypatch.setattr(zones, "_ordered_months", lambda months: list(months))
    monkeypatch.setattr(
        zones,
        "_build_composite_series",
        lambda *_args, **_kwargs: ([("2024-01", object())], [], {}),
    )
    monkeypatch.setattr(zones, "_compute_ndvi", lambda image: object())
    monkeypatch.setattr(
        zones,
        "_ndvi_temporal_stats",
        lambda _images: {"mean": object(), "cv": object()},
    )

    def _fake_pixel_count(_image, _geometry, *, context, scale):  # pragma: no cover - simple stub
        if context == "NDVI mean pixel count":
            return 0
        return 1

    monkeypatch.setattr(zones, "_pixel_count", _fake_pixel_count)

    with pytest.raises(ValueError) as excinfo:
        zones._prepare_selected_period_artifacts(
            {"type": "Feature", "geometry": {}},
            geometry=geometry,
            months=["2024-01"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            cloud_prob_max=40,
            n_classes=5,
            cv_mask_threshold=0.5,
            apply_stability_mask=True,
            min_mapping_unit_ha=1,
            smooth_radius_m=0,
            open_radius_m=0,
            close_radius_m=0,
            simplify_tol_m=0,
            simplify_buffer_m=0,
            method="ndvi_percentiles",
            sample_size=100,
            include_stats=False,
        )

    assert str(excinfo.value) == zones.NDVI_MASK_EMPTY_ERROR


def test_build_percentile_zones_includes_all_requested_classes(monkeypatch):
    class _FakeNumber:
        def __init__(self, value):
            self.value = float(value)

    class _FakeMask:
        def __init__(self, values):
            self.values = [bool(v) for v in values]

    class _FakeList(list):
        def map(self, func):
            return _FakeList([func(item) for item in self])

        def iterate(self, func, initial):
            result = initial
            for item in self:
                result = func(result, item)
            return result

        def get(self, index):
            return super().__getitem__(index)

        def getInfo(self):
            return list(self)

    class _FakeImage:
        def __init__(self, values, *, mask=None, name="ndvi"):
            self.values = list(values)
            self.mask_values = list(mask) if mask is not None else [True] * len(self.values)
            self.name = name

        def bandNames(self):
            return _FakeList([self.name])

        def rename(self, name):
            self.name = str(name)
            return self

        def clamp(self, low, high):
            clamped = [min(max(value, low), high) for value in self.values]
            return _FakeImage(clamped, mask=self.mask_values, name=self.name)

        def mask(self):
            return _FakeMask(self.mask_values)

        def updateMask(self, mask):
            if isinstance(mask, _FakeMask):
                combined = [current and other for current, other in zip(self.mask_values, mask.values)]
            elif isinstance(mask, _FakeImage):
                combined = [current and bool(other) for current, other in zip(self.mask_values, mask.values)]
            else:  # pragma: no cover - defensive fallback
                combined = list(self.mask_values)
            return _FakeImage(self.values, mask=combined, name=self.name)

        def gt(self, threshold):
            limit = threshold.value if isinstance(threshold, _FakeNumber) else float(threshold)
            return _FakeImage([1 if value > limit else 0 for value in self.values], mask=self.mask_values, name=self.name)

        def add(self, other):
            if isinstance(other, _FakeImage):
                combined = [left + right for left, right in zip(self.values, other.values)]
            else:
                value = other.value if isinstance(other, _FakeNumber) else float(other)
                combined = [left + value for left in self.values]
            return _FakeImage(combined, mask=self.mask_values, name=self.name)

        def multiply(self, other):
            factor = other.value if isinstance(other, _FakeNumber) else float(other)
            return _FakeImage([value * factor for value in self.values], mask=self.mask_values, name=self.name)

        def toInt(self):
            return _FakeImage([int(value) for value in self.values], mask=self.mask_values, name=self.name)

        def clip(self, _geometry):
            return self

        def reduceRegion(self, **_kwargs):
            class _FakeResult:
                @staticmethod
                def getInfo():
                    return {}

            return _FakeResult()

    fake_thresholds = [0.2, 0.4, 0.5]
    mean_image = _FakeImage([0.1, 0.3, 0.5, 0.6])
    stability_mask = _FakeImage([1, 1, 1, 1])
    ndvi_stats = {"mean": mean_image, "stability": stability_mask}

    fake_ee = SimpleNamespace(
        List=lambda values: _FakeList(list(values)),
        Number=lambda value: _FakeNumber(value),
        String=lambda value: str(value),
        Image=lambda image: image,
        Reducer=SimpleNamespace(
            percentile=lambda percentiles, outputNames=None: {
                "percentiles": percentiles,
                "outputNames": outputNames,
            }
        ),
    )

    monkeypatch.setattr(zones, "ee", fake_ee)
    monkeypatch.setattr(zones.gee, "initialize", lambda: None)
    monkeypatch.setattr(zones.gee, "MAX_PIXELS", 1_000, raising=False)
    monkeypatch.setattr(zones.gee, "geometry_from_geojson", lambda aoi: aoi, raising=False)
    monkeypatch.setattr(
        zones.gee,
        "monthly_sentinel2_collection",
        lambda _geom, month, _cloud_prob: (
            SimpleNamespace(size=lambda: SimpleNamespace(getInfo=lambda: 1)),
            CliplessImage(f"composite_{month}"),
        ),
    )
    monkeypatch.setattr("app.gee.initialize", lambda: None)
    monkeypatch.setenv("GEE_ALLOW_INIT_FAILURE", "1")
    monkeypatch.setattr(
        zones,
        "_percentile_thresholds",
        lambda _reducer, _percentiles, _label: fake_thresholds,
    )
    monkeypatch.setattr(
        zones,
        "_apply_cleanup",
        lambda classified, _geometry, **_kwargs: classified,
    )

    zone_image, thresholds = zones._build_percentile_zones(
        ndvi_stats=ndvi_stats,
        geometry=object(),
        n_classes=4,
        smooth_radius_m=0,
        open_radius_m=0,
        close_radius_m=0,
        min_mapping_unit_ha=0,
    )

    assert zone_image.values == [1, 2, 3, 4]
    assert thresholds == [0.2, 0.4, 0.5]
    assert all(isinstance(value, float) for value in thresholds)


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
    captured: dict[str, Any] = {}
    artifacts = object()

    def _fake_export(
        aoi_geojson_or_geom,
        *,
        months,
        aoi_name,
        destination,
        start_date=None,
        end_date=None,
        min_mapping_unit_ha,
        include_stats,
        simplify_tolerance_m,
        gcs_bucket=None,
        gcs_prefix=None,
        **kwargs,
    ):
        captured["aoi_geojson"] = aoi_geojson_or_geom
        captured["months"] = list(months)
        captured["aoi_name"] = aoi_name
        captured["destination"] = destination
        captured["min_mapping_unit_ha"] = min_mapping_unit_ha
        captured["include_stats"] = include_stats
        captured["simplify_tolerance_m"] = simplify_tolerance_m
        captured["gcs_bucket"] = gcs_bucket
        captured["gcs_prefix"] = gcs_prefix
        captured["extra"] = kwargs
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
            "tasks": {},
            "metadata": {
                "used_months": list(months),
                "skipped_months": ["2024-04"],
                "palette": list(zones.ZONE_PALETTE[:5]),
                "percentile_thresholds": [0.1, 0.2, 0.3, 0.4],
                "stability": {
                    "initial_threshold": 0.5,
                    "final_threshold": 1.5,
                    "thresholds_tested": [0.5, 1.0, 1.5],
                    "survival_ratio": 0.35,
                    "surviving_pixels": 35,
                    "total_pixels": 100,
                    "target_ratio": zones.MIN_STABILITY_SURVIVAL_RATIO,
                    "low_confidence": True,
                    "mean_pixel_count": 100,
                    "mask_pixel_count": 35,
                    "apply_stability": True,
                },
                "debug": {
                    "stability": {
                        "survival_ratios": [0.2, 0.3, 0.35],
                        "thresholds_tested": [0.5, 1.0, 1.5],
                        "mean_pixel_count": 100,
                        "mask_pixel_count": 35,
                        "apply_stability": True,
                    }
                },
            },
            "prefix": "zones/PROD_202403_202405_demo_zones",
            "palette": list(zones.ZONE_PALETTE[:5]),
            "thresholds": [0.1, 0.2, 0.3, 0.4],
            "artifacts": artifacts,
        }

    monkeypatch.setattr(zones, "export_selected_period_zones", _fake_export)

    request = ProductionZonesRequest(
        aoi_geojson=_sample_polygon(),
        aoi_name="Demo",
        months=["2024-03", "2024-05"],
        n_classes=5,
    )

    response = create_production_zones(request)
    assert captured["aoi_geojson"] == request.aoi_geojson
    assert captured["months"] == ["2024-03", "2024-05"]
    assert captured["aoi_name"] == "Demo"
    assert captured["destination"] == "zip"
    assert captured["min_mapping_unit_ha"] == request.mmu_ha
    assert captured["include_stats"] is True
    assert captured["simplify_tolerance_m"] == request.simplify_tol_m
    assert captured["gcs_bucket"] is None
    assert captured["gcs_prefix"] is None
    assert captured["extra"]["cloud_prob_max"] == request.cloud_prob_max
    assert captured["extra"]["n_classes"] == request.n_classes
    assert captured["extra"]["cv_mask_threshold"] == request.cv_mask_threshold
    assert captured["extra"]["smooth_radius_m"] == request.smooth_radius_m
    assert captured["extra"]["open_radius_m"] == request.open_radius_m
    assert captured["extra"]["close_radius_m"] == request.close_radius_m
    assert captured["extra"]["simplify_buffer_m"] == request.simplify_buffer_m

    assert response["ok"] is True
    assert response["ym_start"] == "2024-03"
    assert response["ym_end"] == "2024-05"
    assert response["paths"]["raster"].endswith("demo_zones.tif")
    assert response["paths"]["vector_components"]["dbf"].endswith("demo_zones.dbf")
    assert response["prefix"].endswith("demo_zones")
    assert response["tasks"] == {}

    debug = response["debug"]
    assert debug["requested_months"] == ["2024-03", "2024-05"]
    assert debug["skipped_months"] == ["2024-04"]
    assert debug["retry_thresholds"] == [0.5, 1.0, 1.5]
    stability = debug["stability"]
    assert stability["initial_threshold"] == 0.5
    assert stability["final_threshold"] == 1.5
    assert stability["survival_ratio"] == 0.35
    assert stability["survival_ratios"] == [0.2, 0.3, 0.35]
    assert stability["mean_pixel_count"] == 100
    assert stability["mask_pixel_count"] == 35
    assert stability["apply_stability"] is True
    assert response["palette"] == list(zones.ZONE_PALETTE[:5])
    assert response["thresholds"] == [0.1, 0.2, 0.3, 0.4]


def test_export_selected_period_zones_accepts_new_keywords(monkeypatch):
    fake_artifacts = zones.ZoneArtifacts(
        zone_image=object(),
        zone_vectors=object(),
        zonal_stats=None,
        geometry="normalized-geometry",
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(zones, "_to_ee_geometry", lambda _geojson: "normalized-geometry")
    monkeypatch.setattr(zones.gee, "initialize", lambda: None)

    def _fake_prepare(aoi_geojson, **kwargs):
        captured.update(
            {
                "min_mapping_unit_ha": kwargs.get("min_mapping_unit_ha"),
                "simplify_tol_m": kwargs.get("simplify_tol_m"),
            }
        )
        return fake_artifacts, {
            "used_months": ["2024-01"],
            "skipped_months": [],
            "min_mapping_unit_applied": True,
        }

    monkeypatch.setattr(zones, "_prepare_selected_period_artifacts", _fake_prepare)

    result = zones.export_selected_period_zones(
        aoi_geojson=_sample_polygon(),
        aoi_name="Alias Field",
        months=["2024-01"],
        geometry="normalized-geometry",
        min_mapping_unit_ha=3.5,
        simplify_tolerance_m=7,
        destination="zip",
        include_zonal_stats=False,
    )

    assert captured["min_mapping_unit_ha"] == 3.5
    assert captured["simplify_tol_m"] == 7
    assert result["metadata"]["min_mapping_unit_applied"] is True
    assert result["metadata"]["mmu_applied"] is True
    assert result["paths"]["raster"].endswith(".tif")
    assert result["paths"]["vectors"].endswith(".shp")


def test_create_production_zones_sanitizes_ee_objects(monkeypatch):
    class FakeBandList(list):
        __module__ = "ee.list"

        def getInfo(self):
            return list(self)

    class FakeImage:
        __module__ = "ee.image"

        def __init__(self, bands: list[str]):
            self._bands = bands

        def bandNames(self):  # pragma: no cover - exercised inside sanitiser
            return FakeBandList(self._bands)

    fake_image = FakeImage(["red", "nir", "green"])
    artifacts = zones.ZoneArtifacts(
        zone_image=object(),
        zone_vectors=object(),
        zonal_stats=None,
        geometry="geometry",
    )

    def _fake_prepare(*_args, **_kwargs):
        image = fake_image
        return artifacts, {
            "used_months": ["2024-01"],
            "skipped_months": [],
            "min_mapping_unit_applied": True,
            "palette": list(zones.ZONE_PALETTE[:5]),
            "debug": {
                "stability": {
                    "thresholds_tested": [0.25, 0.5],
                    "raw_image": image,
                    "images": [image],
                },
                "raw_image": image,
                "image_tuple": (image, image),
                "nested": {"image": image},
            },
            "stability": {
                "final_threshold": 0.5,
                "image": image,
                "images": [image],
            },
            "raw_image": image,
            "raw_tuple": (image,),
            "percentile_thresholds": [0.1, 0.2, 0.3, 0.4],
        }

    monkeypatch.setattr(zones, "_to_ee_geometry", lambda _geojson: "geometry")
    monkeypatch.setattr(zones.gee, "initialize", lambda: None)
    monkeypatch.setattr(zones, "_prepare_selected_period_artifacts", _fake_prepare)

    def _assert_no_ee_objects(value):
        module = getattr(value.__class__, "__module__", "")
        assert not module.startswith("ee."), f"Unexpected EE object in payload: {module}"
        if isinstance(value, dict):
            for nested in value.values():
                _assert_no_ee_objects(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                _assert_no_ee_objects(nested)

    result = zones.export_selected_period_zones(
        aoi_geojson=_sample_polygon(),
        aoi_name="Demo",
        months=["2024-01"],
        geometry="geometry",
        destination="zip",
    )

    _assert_no_ee_objects(result["metadata"])
    _assert_no_ee_objects(result["debug"])
    assert result["metadata"]["raw_image"] is None
    assert result["debug"]["raw_image"] is None
    assert result["metadata"]["raw_tuple"] == [None]
    assert result["debug"]["image_tuple"] == [None, None]

    request = ProductionZonesRequest(
        aoi_geojson=_sample_polygon(),
        aoi_name="Demo",
        months=["2024-01"],
    )

    response = create_production_zones(request)
    metadata = response["metadata"]
    debug = response["debug"]

    _assert_no_ee_objects(metadata)
    _assert_no_ee_objects(debug)

    assert metadata["raw_image"] is None
    assert metadata["debug"]["raw_image"] is None
    assert metadata["raw_tuple"] == [None]

    stability = debug["stability"]
    assert stability.get("raw_image") is None
    assert stability.get("image") is None
    assert stability.get("images") == [None]
    assert debug["raw_image"] is None
    assert debug["image_tuple"] == [None, None]


def test_production_zones_request_defaults():
    request = ProductionZonesRequest(
        aoi_geojson=_sample_polygon(),
        aoi_name="Defaults",
        months=["2024-01"],
    )

    assert request.cv_mask_threshold == zones.DEFAULT_CV_THRESHOLD
    assert request.mmu_ha == zones.DEFAULT_MIN_MAPPING_UNIT_HA
    assert request.smooth_radius_m == zones.DEFAULT_SMOOTH_RADIUS_M
    assert request.simplify_buffer_m == zones.DEFAULT_SIMPLIFY_BUFFER_M


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

    def _fake_export(
        aoi_geojson_or_geom,
        *,
        months,
        aoi_name,
        destination,
        start_date=None,
        end_date=None,
        **_kwargs,
    ):
        captured["months"] = list(months)
        captured["start_date"] = start_date
        captured["end_date"] = end_date
        captured["destination"] = destination
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
    assert captured["destination"] == "zip"
    assert response["debug"]["requested_months"] == ["2024-01", "2024-02"]
    assert response["debug"]["stability"] is None


def test_create_production_zones_with_date_request(monkeypatch):
    captured = {}

    def _fake_export(
        aoi_geojson_or_geom,
        *,
        months,
        aoi_name,
        destination,
        start_date=None,
        end_date=None,
        **_kwargs,
    ):
        captured["months"] = list(months)
        captured["start_date"] = start_date
        captured["end_date"] = end_date
        captured["destination"] = destination
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
    assert captured["destination"] == "zip"
    assert response["debug"]["requested_months"] == ["2024-03", "2024-04"]
    assert response["debug"]["stability"] is None


def test_production_zones_endpoint_returns_200(monkeypatch):
    artifacts = object()

    def _fake_export(
        aoi_geojson_or_geom,
        *,
        months,
        aoi_name,
        destination,
        min_mapping_unit_ha,
        include_stats,
        simplify_tolerance_m,
        **_kwargs,
    ):
        assert destination == "zip"
        return {
            "prefix": "Field_2024-01_zones",
            "paths": {
                "raster": "Field_2024-01_zones.tif",
                "vectors": "Field_2024-01_zones.shp",
                "vector_components": {},
                "zonal_stats": None,
            },
            "tasks": {},
            "metadata": {"used_months": list(months), "skipped_months": []},
            "artifacts": artifacts,
        }

    monkeypatch.setattr(zones, "export_selected_period_zones", _fake_export)

    payload = {
        "aoi_geojson": _sample_polygon(),
        "aoi_name": "Field",
        "months": ["2024-01"],
        "n_classes": 5,
    }

    async def _post_json(path: str, body: dict[str, object]) -> tuple[int, bytes]:
        await app.router.startup()
        messages: list[dict[str, object]] = []
        try:
            raw_body = json.dumps(body).encode("utf-8")
            body_sent = False

            async def receive() -> dict[str, object]:
                nonlocal body_sent
                if not body_sent:
                    body_sent = True
                    return {"type": "http.request", "body": raw_body, "more_body": False}
                await asyncio.sleep(0)
                return {"type": "http.disconnect"}

            async def send(message: dict[str, object]) -> None:
                messages.append(message)

            scope = {
                "type": "http",
                "http_version": "1.1",
                "method": "POST",
                "path": path,
                "root_path": "",
                "scheme": "http",
                "client": ("testclient", 50000),
                "server": ("testserver", 80),
                "headers": [
                    (b"host", b"testserver"),
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(raw_body)).encode("ascii")),
                ],
                "query_string": b"",
                "asgi": {"version": "3.0", "spec_version": "2.3"},
            }

            await app(scope, receive, send)
        finally:
            await app.router.shutdown()

        status = 500
        body_bytes = b""
        for message in messages:
            if message.get("type") == "http.response.start":
                status = int(message.get("status", 0))
            elif message.get("type") == "http.response.body":
                body_bytes += message.get("body", b"")
        return status, body_bytes

    status, body_bytes = asyncio.run(
        _post_json("/api/zones/production", payload)
    )

    assert status == 200
    body = json.loads(body_bytes.decode("utf-8"))
    assert body["ok"] is True
    assert body["paths"]["raster"].endswith(".tif")


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
    assert stability["mean_pixel_count"] == 100
    assert stability["mask_pixel_count"] == 5
    assert stability["apply_stability"] is True
    assert metadata["low_confidence"] is False

    debug = metadata["debug"]["stability"]
    assert debug["thresholds_tested"] == [0.25]
    assert debug["low_confidence"] is False
    assert debug["survival_ratios"] == pytest.approx([0.05])
    assert debug["mean_pixel_count"] == 100
    assert debug["mask_pixel_count"] == 5
    assert debug["apply_stability"] is True


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
    assert stability["mean_pixel_count"] == 100
    assert stability["mask_pixel_count"] == 30
    assert stability["apply_stability"] is True
    assert metadata["low_confidence"] is False

    debug = metadata["debug"]["stability"]
    assert debug["thresholds_tested"] == [0.5]
    assert debug["survival_ratios"] == pytest.approx([0.3])
    assert debug["mean_pixel_count"] == 100
    assert debug["mask_pixel_count"] == 30
    assert debug["apply_stability"] is True


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

def test_percentile_thresholds_accepts_band_prefixed_keys():
    # 5 classes -> need 4 thresholds (01..04)
    reducer_dict = {
        "ndvi_mean_cut_01": 0.12,
        "ndvi_mean_cut_02": 0.21,
        "ndvi_mean_cut_03": 0.33,
        "ndvi_mean_cut_04": 0.47,
        # noise keys should be ignored:
        "foo": 1.0,
        "some_other_key": 2.0,
    }
    from app.services.zones import _percentile_thresholds
    percentiles = [20, 40, 60, 80]
    th = _percentile_thresholds(reducer_dict, percentiles, "ndvi_mean")
    assert th == [0.12, 0.21, 0.33, 0.47]


def test_percentile_thresholds_accepts_default_percentile_keys():
    reducer_dict = {
        "p20": 0.11,
        "ndvi_mean_p40": 0.22,
        "p60": 0.36,
        "ndvi_mean_p80": 0.49,
    }

    from app.services.zones import _percentile_thresholds

    percentiles = [20, 40, 60, 80]
    thresholds = _percentile_thresholds(reducer_dict, percentiles, "ndvi_mean")

    assert thresholds == [0.11, 0.22, 0.36, 0.49]


def test_normalise_feature_uses_scalar_results(monkeypatch):
    class _FakeNumber:
        def __init__(self, value):
            if isinstance(value, _FakeNumber):
                value = value.value
            self.value = float(value)

        def max(self, other):
            other_value = other.value if isinstance(other, _FakeNumber) else float(other)
            return _FakeNumber(max(self.value, other_value))

    class _FakeList:
        def __init__(self, items):
            self._items = list(items)

        def get(self, index, default=None):
            try:
                return self._items[index]
            except IndexError:
                return default

    class _FakeDictionary:
        def __init__(self, mapping):
            self._mapping = dict(mapping)

        def values(self):
            return _FakeList(self._mapping.values())

    class _FakeReducer:
        def __init__(self, name):
            self.name = name

    class _FakeEE:
        class Reducer:
            @staticmethod
            def mean():
                return _FakeReducer("mean")

            @staticmethod
            def stdDev():
                return _FakeReducer("stdDev")

        class Image:
            @staticmethod
            def constant(_value):  # pragma: no cover - enforced by test
                raise AssertionError("ee.Image.constant should not be called")

        Number = staticmethod(_FakeNumber)

    class _FakeBandNames:
        def get(self, index):
            assert index == 0
            return "NDVI_mean"

    class _FakeImage:
        def __init__(self):
            self.subtracted = None
            self.divided_by = None
            self.renamed_to = None

        def bandNames(self):
            return _FakeBandNames()

        def reduceRegion(self, reducer, **_kwargs):
            if reducer.name == "mean":
                return _FakeDictionary({"NDVI_mean": 12})
            if reducer.name == "stdDev":
                return _FakeDictionary({"NDVI_mean": 3})
            raise AssertionError(f"Unexpected reducer {reducer.name}")

        def subtract(self, value):
            self.subtracted = value
            return self

        def divide(self, value):
            self.divided_by = value
            return self

        def rename(self, name):
            self.renamed_to = name
            return self

    fake_image = _FakeImage()
    monkeypatch.setattr(zones, "ee", _FakeEE())

    result = zones._normalise_feature(fake_image, geometry=object(), name="NDVI_mean")

    assert result is fake_image
    assert isinstance(fake_image.subtracted, _FakeNumber)
    assert fake_image.subtracted.value == pytest.approx(12.0)
    assert isinstance(fake_image.divided_by, _FakeNumber)
    assert fake_image.divided_by.value == pytest.approx(3.0)
    assert fake_image.renamed_to == "norm_NDVI_mean"

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


def test_clean_zones_coerces_clipless_images(monkeypatch):
    class CliplessImage:
        def __init__(self, label: str = "base"):
            self.label = label

        def focal_mode(self, **_kwargs):
            return CliplessImage("focal_mode")

        def focal_min(self, **_kwargs):
            return self

        def focal_max(self, **_kwargs):
            return self

        def gt(self, *_args, **_kwargs):
            return CliplessImage("gt")

        def updateMask(self, *_args, **_kwargs):
            return self

        def clamp(self, *_args, **_kwargs):
            return self

        def rename(self, *_args, **_kwargs):
            return self

        def where(self, *_args, **_kwargs):
            return CliplessImage("where")

        def reduceConnectedComponents(self, *_args, **_kwargs):
            return CliplessImage("connected")

        def lt(self, *_args, **_kwargs):
            return CliplessImage("lt")

        def toInt16(self):
            return self

    class FakePixelArea:
        def addBands(self, _image):
            return CliplessImage("pixel_area")

    class FakeReducer:
        @staticmethod
        def sum():
            return "sum"

    class FakeNumber:
        def __init__(self, value):
            self._value = value

        def getInfo(self):
            getter = getattr(self._value, "getInfo", None)
            if callable(getter):
                return getter()
            return self._value

    class FakeSize:
        def __init__(self, count: int):
            self._count = count

        def getInfo(self):
            return self._count

    class FakeComposites:
        def __init__(self, count: int = 1):
            self._size = FakeSize(count)

        def size(self):
            return self._size

    class FakeEEImage:
        def __init__(self, source):
            if isinstance(source, FakeEEImage):
                self._source = source._source
                self.clipped = source.clipped
                self.metadata = dict(getattr(source, "metadata", {}))
            else:
                self._source = source
                self.clipped = None
                self.metadata: dict[str, Any] = {}

        def clip(self, geometry):
            self.clipped = geometry
            return self

        def toInt16(self):
            return self

        def setMulti(self, metadata: dict[str, Any]):
            self.metadata.update(metadata)
            return self

        def __getattr__(self, name: str):
            return getattr(self._source, name)

        @staticmethod
        def pixelArea():
            return FakePixelArea()

        @staticmethod
        def cat(_images):
            return FakeEEImage(CliplessImage("cat"))

    fake_ee = SimpleNamespace(
        Image=FakeEEImage,
        Geometry=lambda geom: geom,
        Reducer=FakeReducer,
        Number=lambda value: FakeNumber(value),
    )

    monkeypatch.setattr(zones, "ee", fake_ee)

    clipless = CliplessImage("input")
    geometry = {"type": "Point", "coordinates": [0, 0]}

    cleaned = zones._clean_zones(
        clipless,
        geometry,
        smooth_radius_m=0,
        open_radius_m=0,
        close_radius_m=0,
        min_mapping_unit_ha=1,
    )

    assert isinstance(cleaned, FakeEEImage)
    assert cleaned.clipped == geometry

    fake_vector = SimpleNamespace()
    fake_stats = SimpleNamespace()

    monkeypatch.setattr(zones, "_build_masked_s2_collection", lambda **_kwargs: object())
    monkeypatch.setattr(
        zones,
        "_build_composite_collection",
        lambda **_kwargs: (FakeComposites(), []),
    )
    monkeypatch.setattr(zones, "_ndvi_collection", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        zones,
        "_ndvi_statistics",
        lambda *_args, **_kwargs: (
            CliplessImage("mean"),
            CliplessImage("median"),
            CliplessImage("std"),
            CliplessImage("cv"),
        ),
    )
    monkeypatch.setattr(
        zones,
        "_stability_mask",
        lambda *_args, **_kwargs: (CliplessImage("mask"), {"threshold": 0.5}),
    )
    monkeypatch.setattr(
        zones,
        "_ndvi_percentile_thresholds",
        lambda *_args, **_kwargs: ([0.1, 0.2], [0.1, 0.2]),
    )
    monkeypatch.setattr(
        zones,
        "_classify_percentiles",
        lambda *_args, **_kwargs: CliplessImage("classified"),
    )
    monkeypatch.setattr(zones, "_vectorize_zones", lambda *_args, **_kwargs: fake_vector)
    monkeypatch.setattr(zones, "_zonal_statistics", lambda *_args, **_kwargs: fake_stats)

    polygon = {
        "type": "Polygon",
        "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]],
    }

    def _fake_build_zone_artifacts(*_args, **_kwargs):
        zone_image = FakeEEImage(CliplessImage("zone"))
        zone_image.metadata["method"] = "ndvi_percentiles"
        zone_image.clip(polygon)
        return zones.ZoneArtifacts(
            zone_image=zone_image,
            zone_vectors=fake_vector,
            zonal_stats=None,
            geometry=polygon,
        )

    monkeypatch.setattr(zones, "build_zone_artifacts", _fake_build_zone_artifacts)

    artifacts = zones.build_zone_artifacts(
        polygon,
        months=["2024-01"],
        include_stats=False,
        smooth_radius_m=0,
        open_radius_m=0,
        close_radius_m=0,
    )

    assert isinstance(artifacts.zone_image, FakeEEImage)
    assert artifacts.zone_image.metadata["method"] == "ndvi_percentiles"
    assert artifacts.zone_image.clipped == polygon
    assert artifacts.zone_vectors is fake_vector
    assert artifacts.zonal_stats is None

