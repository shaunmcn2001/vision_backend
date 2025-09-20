import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

from app.api import export
from app.services import tiles


class FakeNDVIImage:
    def __init__(self, value: float):
        self.value = value
        self.name = None

    def rename(self, name: str):
        self.name = name
        return self


class FakeImage:
    def __init__(self, ndvi_value: float):
        self._raw_ndvi = ndvi_value
        self.ndvi_value = None

    def normalizedDifference(self, _bands):
        return FakeNDVIImage(self._raw_ndvi)

    def addBands(self, ndvi_img):
        self.ndvi_value = getattr(ndvi_img, "value", None)
        return self


class FakeMeanImage:
    def __init__(self, value: float):
        self.value = value
        self.clamped_to = None
        self.clipped_geom = None
        self.resample_method = None
        self.reproject_args = None

    def resample(self, method: str):
        self.resample_method = method
        return self

    def reproject(self, crs: str, transform, scale: int):
        self.reproject_args = (crs, transform, scale)
        return self

    def clip(self, geom):
        self.clipped_geom = geom
        return self

    def clamp(self, low, high):
        self.clamped_to = (low, high)
        if self.value is None:
            return self
        if self.value < low:
            self.value = low
        elif self.value > high:
            self.value = high
        return self


class FakeImageCollection:
    def __init__(self, name: str, context: dict):
        self.name = name
        self.context = context
        self.geom = None
        self.start = None
        self.end = None
        self.filters = []
        self.images = [FakeImage(val) for val in context["values"]]

    def filterBounds(self, geom):
        self.geom = geom
        return self

    def filterDate(self, start, end):
        self.start = start
        self.end = end
        return self

    def filter(self, filt):
        self.filters.append(filt)
        return self

    def map(self, func):
        self.images = [func(img) for img in self.images]
        return self

    def select(self, _band):
        return self

    def mean(self):
        values = [img.ndvi_value if img.ndvi_value is not None else img._raw_ndvi for img in self.images]
        mean_val = sum(values) / len(values) if values else None
        return FakeMeanImage(mean_val)


def _setup_fake_ee(monkeypatch, module, values):
    context = {"values": list(values)}

    def fake_image_collection(name):
        context_copy = {"values": list(context["values"])}
        return FakeImageCollection(name, context_copy)

    fake_filter = SimpleNamespace(lt=lambda *args, **kwargs: ("lt", args, kwargs))
    fake_ee = SimpleNamespace(
        ImageCollection=fake_image_collection,
        Geometry=lambda geom: geom,
        Filter=fake_filter,
    )

    monkeypatch.setattr(module, "ee", fake_ee)

    return context


def test_export_ndvi_range_preserves_negatives(monkeypatch):
    context = _setup_fake_ee(monkeypatch, export, [-0.6, -0.2])

    _, image = export._ndvi_image_for_range({"type": "Point", "coordinates": [0, 0]}, "2024-01-01", "2024-02-01")

    assert isinstance(image, FakeMeanImage)
    assert image.clamped_to == (-1, 1)
    assert image.value == pytest.approx(-0.4)
    assert image.resample_method == "bilinear"
    assert image.reproject_args == ("EPSG:3857", None, 10)
    assert image.clipped_geom == {"type": "Point", "coordinates": [0, 0]}

    # Updating context should not affect the already computed image
    context["values"] = [-0.1, 0.2]
    assert image.value == pytest.approx(-0.4)


def test_tile_ndvi_images_allow_negative_values(monkeypatch):
    context = _setup_fake_ee(monkeypatch, tiles, [-0.8, -0.4])

    annual = tiles.ndvi_annual_image({"type": "Polygon", "coordinates": []}, 2021)
    assert isinstance(annual, FakeMeanImage)
    assert annual.clamped_to == (-1, 1)
    assert annual.value == pytest.approx(-0.6)
    assert annual.resample_method == "bilinear"
    assert annual.reproject_args is None
    assert annual.clipped_geom == {"type": "Polygon", "coordinates": []}

    context["values"] = [-0.7, 0.1]
    month = tiles.ndvi_month_image({"type": "Polygon", "coordinates": []}, 2021, 5)
    assert isinstance(month, FakeMeanImage)
    assert month.clamped_to == (-1, 1)
    assert month.value == pytest.approx(-0.3)
    assert month.resample_method == "bilinear"
    assert month.reproject_args is None
    assert month.clipped_geom == {"type": "Polygon", "coordinates": []}
    assert annual.value == pytest.approx(-0.6)
