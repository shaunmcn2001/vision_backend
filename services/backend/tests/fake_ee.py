from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Iterable


class FakeIndexBand:
    def __init__(self, value: float, context: dict[str, Any]):
        self.value = value
        self.name: str | None = None
        self._context = context

    def rename(self, name: str):
        self.name = name
        log = self._context.setdefault("log", {})
        log.setdefault("renamed_bands", []).append(name)
        return self


class FakeImage:
    def __init__(self, raw_value: float, context: dict[str, Any]):
        self._raw_value = raw_value
        self._context = context
        self.index_value: float | None = None

    def normalizedDifference(self, bands: Iterable[str]):
        log = self._context.setdefault("log", {})
        log.setdefault("normalized_difference_bands", []).append(tuple(bands))
        return FakeIndexBand(self._raw_value, self._context)

    def addBands(self, band_image: FakeIndexBand):
        self.index_value = getattr(band_image, "value", None)
        log = self._context.setdefault("log", {})
        log.setdefault("added_bands", []).append(getattr(band_image, "name", None))
        return self

    def select(self, band: str):
        log = self._context.setdefault("log", {})
        log.setdefault("selected_bands", []).append(band)
        return self


class FakeMeanImage:
    def __init__(self, value: float | None):
        self.value = value
        self.clamped_to: tuple[float, float] | None = None
        self.clipped_geom: Any = None
        self.requested_vis: dict[str, Any] | None = None

    def clip(self, geom: Any):
        self.clipped_geom = geom
        return self

    def clamp(self, low: float, high: float):
        self.clamped_to = (low, high)
        if self.value is None:
            return self
        if self.value < low:
            self.value = low
        elif self.value > high:
            self.value = high
        return self

    def getMapId(self, vis: dict[str, Any]):
        self.requested_vis = vis
        return {
            "mapid": "fake-mapid",
            "token": "fake-token",
            "tile_fetcher": SimpleNamespace(url_format="https://fake/{z}/{x}/{y}"),
        }


class FakeImageCollection:
    def __init__(self, name: str, context: dict[str, Any]):
        self.name = name
        self.context = context
        self.geom: Any = None
        self.start: str | None = None
        self.end: str | None = None
        self.filters: list[Any] = []
        values = context.get("values", [])
        self.images = [FakeImage(val, context) for val in values]
        context.setdefault("log", {}).setdefault("collections", []).append(name)

    def filterBounds(self, geom: Any):
        self.geom = geom
        self.context.setdefault("log", {})["geometry"] = geom
        return self

    def filterDate(self, start: str, end: str):
        self.start = start
        self.end = end
        self.context.setdefault("log", {})["date_range"] = (start, end)
        return self

    def filter(self, filt: Any):
        self.filters.append(filt)
        self.context.setdefault("log", {}).setdefault("filters", []).append(filt)
        return self

    def map(self, func):
        self.images = [func(img) for img in self.images]
        return self

    def select(self, band: str):
        self.context.setdefault("log", {}).setdefault("selected_bands", []).append(band)
        return self

    def mean(self):
        values = [
            img.index_value if img.index_value is not None else img._raw_value
            for img in self.images
        ]
        mean_val = sum(values) / len(values) if values else None
        return FakeMeanImage(mean_val)


def setup_fake_ee(monkeypatch, module, values: Iterable[float]):
    context: dict[str, Any] = {"values": list(values), "log": {}}

    def fake_image_collection(name: str):
        context_copy = {"values": list(context["values"]), "log": context["log"]}
        return FakeImageCollection(name, context_copy)

    fake_filter = SimpleNamespace(
        lt=lambda *args, **kwargs: ("lt", args, kwargs),
        calendarRange=lambda start, _end, _unit: start,
    )

    fake_ee = SimpleNamespace(
        ImageCollection=fake_image_collection,
        Geometry=lambda geom: geom,
        Filter=fake_filter,
    )

    monkeypatch.setattr(module, "ee", fake_ee)

    return context


__all__ = [
    "FakeImage",
    "FakeImageCollection",
    "FakeIndexBand",
    "FakeMeanImage",
    "setup_fake_ee",
]
