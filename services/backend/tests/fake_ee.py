from __future__ import annotations

from math import sqrt
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

    def select(self, band: Any):
        log = self._context.setdefault("log", {})
        if isinstance(band, (list, tuple)):
            entry = band[0] if len(band) == 1 else tuple(band)
        else:
            entry = band
        log.setdefault("selected_bands", []).append(entry)
        return self

    def toFloat(self):
        log = self._context.setdefault("log", {})
        log.setdefault("to_float_calls", 0)
        log["to_float_calls"] += 1
        return self

    def rename(self, name: str):
        log = self._context.setdefault("log", {})
        log.setdefault("renamed_images", []).append(name)
        return self


class FakeMeanImage:
    def __init__(self, value: float | None, context: dict[str, Any] | None = None):
        self.value = value
        self.clamped_to: tuple[float, float] | None = None
        self.clipped_geom: Any = None
        self.requested_vis: dict[str, Any] | None = None
        self.download_args: dict[str, Any] | None = None
        self._context = context
        self.name: str | None = None
        self.mask: Any = None

    def _wrap(self, value: float | None):
        return FakeMeanImage(value, self._context)

    @staticmethod
    def _value_of(other: Any) -> float | None:
        if isinstance(other, FakeMeanImage):
            return other.value
        return other

    def clip(self, geom: Any):
        self.clipped_geom = geom
        return self

    def clamp(self, low: float, high: float):
        self.clamped_to = (low, high)
        if self._context is not None:
            log = self._context.setdefault("log", {})
            log.setdefault("clamp_calls", []).append((low, high))
        if self.value is None:
            return self
        if self.value < low:
            self.value = low
        elif self.value > high:
            self.value = high
        return self

    def rename(self, name: str):
        self.name = name
        if self._context is not None:
            log = self._context.setdefault("log", {})
            log.setdefault("renamed_images", []).append(name)
        return self

    def updateMask(self, mask: Any):
        self.mask = mask
        if self._context is not None:
            log = self._context.setdefault("log", {})
            log.setdefault("update_masks", []).append(mask)
        return self

    def divide(self, other: Any):
        other_value = self._value_of(other)
        if other_value in (0, None) or self.value is None:
            return self._wrap(None)
        return self._wrap(self.value / other_value)

    def where(self, condition: Any, substitute: Any):
        cond_value = self._value_of(condition)
        if cond_value:
            sub_value = self._value_of(substitute)
            return self._wrap(sub_value)
        return self._wrap(self.value)

    def gt(self, other: Any):
        other_value = self._value_of(other)
        if self.value is None or other_value is None:
            return self._wrap(0)
        return self._wrap(1 if self.value > other_value else 0)

    def eq(self, other: Any):
        other_value = self._value_of(other)
        if self.value is None or other_value is None:
            return self._wrap(0)
        return self._wrap(1 if self.value == other_value else 0)

    def lt(self, other: Any):
        other_value = self._value_of(other)
        if self.value is None or other_value is None:
            return self._wrap(0)
        return self._wrap(1 if self.value < other_value else 0)

    def lte(self, other: Any):
        other_value = self._value_of(other)
        if self.value is None or other_value is None:
            return self._wrap(0)
        return self._wrap(1 if self.value <= other_value else 0)

    def abs(self):
        if self.value is None:
            return self._wrap(None)
        return self._wrap(abs(self.value))

    def getDownloadURL(self, params: dict[str, Any]):
        self.download_args = params
        if self._context is not None:
            log = self._context.setdefault("log", {})
            log.setdefault("download_args", []).append(params)
        return "https://example.com/download"

    def getMapId(self, vis: dict[str, Any]):
        self.requested_vis = vis
        return {
            "mapid": "fake-mapid",
            "token": "fake-token",
            "tile_fetcher": SimpleNamespace(url_format="https://fake/{z}/{x}/{y}"),
        }


class FakeImageCollection:
    def __init__(
        self,
        name: str,
        context: dict[str, Any],
        images: Iterable[Any] | None = None,
    ):
        self.name = name
        self.context = context
        self.geom: Any = None
        self.start: str | None = None
        self.end: str | None = None
        self.filters: list[Any] = []
        if images is None:
            values = context.get("values", [])
            self.images = [FakeImage(val, context) for val in values]
        else:
            self.images = list(images)
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
        return FakeMeanImage(mean_val, self.context)

    def size(self):
        count = len(self.images)
        return SimpleNamespace(getInfo=lambda: count)

    def reduce(self, reducer: Any):
        kind = getattr(reducer, "kind", str(reducer))
        log = self.context.setdefault("log", {})
        log.setdefault("reduce_calls", []).append(kind)

        def _value(item: Any) -> float | None:
            if isinstance(item, FakeMeanImage):
                return item.value
            if hasattr(item, "index_value") and item.index_value is not None:
                return item.index_value
            return getattr(item, "_raw_value", None)

        raw_values = [_value(img) for img in self.images]
        values = [val for val in raw_values if val is not None]

        if kind == "sum":
            result = sum(values) if values else 0
        elif kind == "count":
            result = len(values)
        elif kind == "mean":
            result = sum(values) / len(values) if values else None
        elif kind == "median":
            if not values:
                result = None
            else:
                ordered = sorted(values)
                mid = len(ordered) // 2
                if len(ordered) % 2:
                    result = ordered[mid]
                else:
                    result = (ordered[mid - 1] + ordered[mid]) / 2
        elif kind == "stdDev":
            if not values:
                result = 0
            else:
                mean_val = sum(values) / len(values)
                variance = sum((val - mean_val) ** 2 for val in values) / len(values)
                result = sqrt(variance)
        else:
            result = None

        return FakeMeanImage(result, self.context)


def setup_fake_ee(monkeypatch, module, values: Iterable[float]):
    context: dict[str, Any] = {"values": list(values), "log": {}}

    def fake_image_collection(source: Any):
        if isinstance(source, FakeImageCollection):
            return source

        context_copy = {"values": list(context["values"]), "log": context["log"]}
        if isinstance(source, (list, tuple)):
            return FakeImageCollection("inline", context_copy, images=list(source))
        return FakeImageCollection(source, context_copy)

    try:
        from app.services import image_stats as image_stats_module  # noqa: WPS433
    except Exception:  # pragma: no cover - defensive
        image_stats_module = None

    modules: list[Any]
    if isinstance(module, (list, tuple, set)):
        modules = list(module)
    else:
        modules = [module]

    if image_stats_module is not None and image_stats_module not in modules:
        modules.append(image_stats_module)

    def _make_fake_filter():
        return SimpleNamespace(
            lt=lambda *args, **kwargs: ("lt", args, kwargs),
            calendarRange=lambda start, _end, _unit: start,
        )

    class FakeReducer:
        def __init__(self, kind: str):
            self.kind = kind

    def _make_fake_reducer():
        return SimpleNamespace(
            sum=lambda: FakeReducer("sum"),
            count=lambda: FakeReducer("count"),
            median=lambda: FakeReducer("median"),
            stdDev=lambda: FakeReducer("stdDev"),
            mean=lambda: FakeReducer("mean"),
        )

    class FakeImageFactory:
        def __init__(self, ctx: dict[str, Any]):
            self._ctx = ctx

        def __call__(self, value: Any):
            return value

        def constant(self, value: float):
            return FakeMeanImage(value, self._ctx)

    for target in modules:
        fake_image = FakeImageFactory(context)
        fake_ee = SimpleNamespace(
            ImageCollection=fake_image_collection,
            Geometry=lambda geom: geom,
            Filter=_make_fake_filter(),
            Reducer=_make_fake_reducer(),
            Image=fake_image,
        )
        monkeypatch.setattr(target, "ee", fake_ee)

    return context


__all__ = [
    "FakeImage",
    "FakeImageCollection",
    "FakeIndexBand",
    "FakeMeanImage",
    "setup_fake_ee",
]
