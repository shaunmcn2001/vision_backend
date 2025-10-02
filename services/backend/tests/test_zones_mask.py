from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import ee
import pytest

from app import gee
from app.services import zones

_AOI_COORDS = [
    [149.0, -35.0],
    [149.001, -35.0],
    [149.001, -35.001],
    [149.0, -35.001],
    [149.0, -35.0],
]


def _geometry() -> ee.Geometry:
    return ee.Geometry.Polygon([_AOI_COORDS])


def _run_case(start: date, end: date) -> tuple[int, int, int]:
    geometry = _geometry()
    months = zones._months_from_dates(start, end)
    composites, _skipped, _meta = zones._build_composite_series(
        geometry,
        months,
        start,
        end,
        zones.DEFAULT_CLOUD_PROB_MAX,
    )
    if not composites:
        pytest.skip("No Sentinel-2 composites available for test window")

    ndvi_images = [zones._compute_ndvi(image) for _, image in composites]
    stats = zones._ndvi_temporal_stats(ndvi_images)
    cv_image = stats["cv"]

    thresholds = [zones.DEFAULT_CV_THRESHOLD]
    for fallback in zones.STABILITY_THRESHOLD_SEQUENCE:
        if fallback > zones.DEFAULT_CV_THRESHOLD + 1e-9:
            thresholds.append(fallback)

    stability_mask = zones._stability_mask(
        cv_image,
        geometry,
        thresholds,
        zones.MIN_STABILITY_SURVIVAL_RATIO,
        zones.DEFAULT_SCALE,
    )

    mean_count = zones._pixel_count(
        stats["mean"],
        geometry,
        context="zones test mean count",
        scale=zones.DEFAULT_SCALE,
    )
    mask_count = zones._pixel_count(
        stability_mask,
        geometry,
        context="zones test stability count",
        scale=zones.DEFAULT_SCALE,
    )

    ndvi_stats = {**stats, "stability": stability_mask}
    zone_image, _thresholds = zones._build_percentile_zones(
        ndvi_stats=ndvi_stats,
        geometry=geometry,
        n_classes=zones.DEFAULT_N_CLASSES,
        smooth_radius_m=zones.DEFAULT_SMOOTH_RADIUS_M,
        open_radius_m=zones.DEFAULT_OPEN_RADIUS_M,
        close_radius_m=zones.DEFAULT_CLOSE_RADIUS_M,
        min_mapping_unit_ha=0,
    )
    zone_count = zones._pixel_count(
        zone_image.updateMask(zone_image),
        geometry,
        context="zones test zone count",
        scale=zones.DEFAULT_SCALE,
    )

    return mean_count, mask_count, zone_count


@pytest.fixture(scope="module")
def ee_ready():
    try:
        gee.initialize()
    except RuntimeError:
        pytest.skip("Earth Engine credentials not configured")
    zones.set_apply_stability(True)
    return True


def test_stability_mask_counts_short_range(ee_ready):
    mean_count, mask_count, zone_count = _run_case(
        start=date(2023, 1, 1),
        end=date(2023, 3, 31),
    )

    assert mean_count > 0
    assert mask_count > 0
    assert zone_count > 0


def test_stability_mask_counts_long_range(ee_ready):
    mean_count, mask_count, zone_count = _run_case(
        start=date(2022, 1, 1),
        end=date(2023, 12, 31),
    )

    assert mean_count > 0
    assert mask_count > 0
    assert zone_count > 0


def test_stability_mask_fallback_prevents_empty_error(monkeypatch):
    total_pixels = 5
    threshold_values = [zones.DEFAULT_CV_THRESHOLD]
    threshold_values.extend(
        fallback
        for fallback in zones.STABILITY_THRESHOLD_SEQUENCE
        if fallback > zones.DEFAULT_CV_THRESHOLD + 1e-9
    )
    survivors_by_threshold = {round(value, 6): 0 for value in threshold_values}

    class _FakeListValues(list):
        def get(self, index):
            return self[index]

    class _FakeDict:
        def __init__(self, value: int):
            self._value = int(value)

        def values(self):
            return _FakeListValues([self._value])

    def _coerce(value) -> float:
        if isinstance(value, _FakeNumber):
            return value.value
        return float(value)

    class _FakeNumber:
        def __init__(self, value):
            self.value = float(value or 0)

        def divide(self, other):
            other_val = _coerce(other)
            if other_val == 0:
                return _FakeNumber(0)
            return _FakeNumber(self.value / other_val)

        def max(self, other):
            return _FakeNumber(max(self.value, _coerce(other)))

        def gte(self, other):
            return self.value >= _coerce(other)

        def lte(self, other):
            return self.value <= _coerce(other)

        def __bool__(self):
            return bool(self.value)

    class _FakeList(list):
        def map(self, func):
            return [func(item) for item in self]

    class _FakeMask:
        def __init__(self, total: int, survivors: int, label: str):
            self.total = int(total)
            self.survivors = int(survivors)
            self.label = label
            self._masked = False
            self.is_pass_through = False

        def selfMask(self):
            self._masked = True
            return self

        def updateMask(self, *_args, **_kwargs):
            return self

        def reduceRegion(self, **_kwargs):
            count = self.survivors if self._masked else self.total
            return _FakeDict(count)

    class _FakeImageCollection:
        def __init__(self, images):
            self.images = list(images)

        def max(self):
            if not self.images:
                return _FakeMask(0, 0, "empty")
            total = self.images[0].total if self.images else 0
            survivors = 0
            is_pass = False
            for image in self.images:
                if isinstance(image, _FakeMask):
                    current = image.survivors if image._masked else image.total
                    survivors = max(survivors, current)
                    is_pass = is_pass or getattr(image, "is_pass_through", False)
            mask = _FakeMask(total, survivors, "combined")
            mask.is_pass_through = is_pass
            return mask

    class _FakeCVImage:
        def __init__(self, total: int, survivors_map: dict[float, int]):
            self.total = int(total)
            self._survivors = survivors_map

        def reduceRegion(self, **_kwargs):
            return _FakeDict(self.total)

        def lte(self, threshold):
            value = getattr(threshold, "value", threshold)
            key = round(float(value), 6)
            survivors = self._survivors.get(key, 0)
            return _FakeMask(self.total, survivors, f"lte_{key}")

    def _make_number(value):
        if isinstance(value, _FakeNumber):
            return _FakeNumber(value.value)
        return _FakeNumber(value or 0)

    def _make_image(value):
        if isinstance(value, _FakeMask):
            return value
        if isinstance(value, _FakeNumber):
            value = value.value
        mask = _FakeMask(total_pixels, total_pixels if value else 0, f"const_{value}")
        if value:
            mask.is_pass_through = True
        return mask

    fake_ee = SimpleNamespace(
        Number=_make_number,
        List=lambda values: _FakeList(values),
        Image=_make_image,
        ImageCollection=SimpleNamespace(fromImages=lambda images: _FakeImageCollection(images)),
        Geometry=lambda geojson: geojson,
        Reducer=SimpleNamespace(count=lambda: object()),
        Algorithms=SimpleNamespace(
            If=lambda condition, truthy, falsey: truthy if bool(condition) else falsey
        ),
    )

    monkeypatch.setattr(zones, "ee", fake_ee)
    monkeypatch.setattr(zones, "gee", SimpleNamespace(MAX_PIXELS=1_000, initialize=lambda: None))

    geometry = object()
    cv_image = _FakeCVImage(total_pixels, survivors_by_threshold)

    mask = zones._stability_mask(
        cv_image,
        geometry,
        threshold_values,
        zones.MIN_STABILITY_SURVIVAL_RATIO,
        zones.DEFAULT_SCALE,
    )

    mask_count = mask.reduceRegion(
        reducer=zones.ee.Reducer.count(),
        geometry=geometry,
        scale=zones.DEFAULT_SCALE,
        bestEffort=True,
        tileScale=4,
        maxPixels=zones.gee.MAX_PIXELS,
    ).values().get(0)

    assert mask_count == total_pixels
    assert getattr(mask, "is_pass_through", False)

    class _FakeZoneImage:
        def __init__(self, properties: dict[str, object]):
            self._properties = properties

        def updateMask(self, *_args, **_kwargs):
            return self

        def neq(self, _value):
            return self

        def toInt16(self):
            return self

        def rename(self, _name):
            return self

        def get(self, name):
            if name not in self._properties:
                return None
            value = self._properties[name]

            class _InfoWrapper:
                def __init__(self, info):
                    self._info = info

                def getInfo(self):  # pragma: no cover - trivial
                    return self._info

            return _InfoWrapper(value)

    class _FakeZoneVectors:
        pass

    def _fake_build_zone_artifacts(
        aoi_geojson_or_geom,
        *,
        months,
        include_stats,
        **_kwargs,
    ):
        assert months == ["2024-01"]
        assert include_stats is False

        thresholds_to_try = [zones.DEFAULT_CV_THRESHOLD]
        thresholds_to_try.extend(
            fallback
            for fallback in zones.STABILITY_THRESHOLD_SEQUENCE
            if fallback > zones.DEFAULT_CV_THRESHOLD + 1e-9
        )
        stability = zones._stability_mask(
            _FakeCVImage(total_pixels, survivors_by_threshold),
            geometry,
            thresholds_to_try,
            zones.MIN_STABILITY_SURVIVAL_RATIO,
            zones.DEFAULT_SCALE,
        )
        stability_count = stability.reduceRegion(
            reducer=zones.ee.Reducer.count(),
            geometry=geometry,
            scale=zones.DEFAULT_SCALE,
            bestEffort=True,
            tileScale=4,
            maxPixels=zones.gee.MAX_PIXELS,
        ).values().get(0)
        if stability_count <= 0:
            raise ValueError(zones.STABILITY_MASK_EMPTY_ERROR)

        zone_image = _FakeZoneImage(
            {
                "months_used": months,
                "months_skipped": [],
                "thresholds": [0.2, 0.4, 0.6],
                "palette": ["#123456", "#abcdef"],
                "stability": {"thresholds_tested": thresholds_to_try},
            }
        )

        return zones.ZoneArtifacts(
            zone_image=zone_image,
            zone_vectors=_FakeZoneVectors(),
            zonal_stats=None,
            geometry=geometry,
        )

    monkeypatch.setattr(zones, "build_zone_artifacts", _fake_build_zone_artifacts)

    result = zones.export_selected_period_zones(
        aoi_geojson={"type": "Polygon", "coordinates": []},
        aoi_name="Demo Field",
        months=["2024-01"],
        geometry=geometry,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
        cv_mask_threshold=zones.DEFAULT_CV_THRESHOLD,
        include_zonal_stats=False,
        export_target="zip",
    )

    assert result["metadata"]["used_months"] == ["2024-01"]
    assert result["thresholds"] == [0.2, 0.4, 0.6]
    assert isinstance(result["artifacts"].zone_image, _FakeZoneImage)
