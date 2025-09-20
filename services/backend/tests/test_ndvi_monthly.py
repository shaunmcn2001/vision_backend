import sys
from pathlib import Path
from types import SimpleNamespace

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

from app.api import routes
from app.services import ndvi


class FakeValue:
    def __init__(self, val):
        self._val = val

    def getInfo(self):
        return self._val


class FakeRegionResult:
    def __init__(self, month):
        self._month = month

    def get(self, key):
        return FakeValue(self._month / 100)


class FakeNumber:
    def __init__(self, value):
        self._value = value

    def eq(self, other):
        return FakeValue(self._value == other)

    def getInfo(self):
        return self._value


class FakeImage:
    reduce_region_calls: list[dict] = []

    def __init__(self, month):
        self._month = month

    def select(self, band):
        return self

    def reduceRegion(self, *args, **kwargs):
        FakeImage.reduce_region_calls.append(kwargs.copy())
        return FakeRegionResult(self._month)


class FakeFilteredCollection:
    def __init__(self, month, *, size: int = 1):
        self._month = month
        self._size = size

    def size(self):
        return FakeNumber(self._size)

    def mean(self):
        if self._size == 0:
            raise RuntimeError("mean() on empty collection")
        return FakeImage(self._month)


class FakeMappedCollection:
    def __init__(self, parent):
        self._parent = parent

    def filter(self, month):
        size = 0 if month in getattr(self._parent, "empty_months", set()) else 1
        return FakeFilteredCollection(month, size=size)


class FakeImageCollection:
    def __init__(self, name):
        self.name = name
        self.geom = None
        self.start = None
        self.end = None
        self.empty_months: set[int] = set()

    def filterBounds(self, geom):
        self.geom = geom
        return self

    def filterDate(self, start, end):
        self.start = start
        self.end = end
        return self

    def map(self, func):
        return FakeMappedCollection(self)


class FakeSequence(list):
    def __init__(self, start, end):
        super().__init__(range(start, end + 1))

    def getInfo(self):
        return list(self)


def test_ndvi_monthly_defaults(monkeypatch):
    captured = {}
    FakeImage.reduce_region_calls = []

    def fake_image_collection(name):
        captured["collection"] = name
        return FakeImageCollection(name)

    fake_ee = SimpleNamespace(
        ImageCollection=fake_image_collection,
        Geometry=lambda geom: geom,
        Filter=SimpleNamespace(calendarRange=lambda start, end, unit: start),
        List=SimpleNamespace(sequence=lambda start, end: FakeSequence(start, end)),
        Reducer=SimpleNamespace(mean=lambda: "mean"),
    )

    monkeypatch.setattr(routes, "ee", fake_ee)
    monkeypatch.setattr(routes, "init_ee", lambda: None)

    request = routes.NDVIRequest(
        geometry={"type": "Point", "coordinates": [0, 0]},
        start="2023-01-01",
        end="2023-12-31",
    )

    data = routes.ndvi_monthly(request)

    assert data["ok"] is True
    assert len(data["data"]) == 12
    assert data["data"][0] == {"month": 1, "ndvi": 0.01}
    assert captured["collection"] == "COPERNICUS/S2_SR_HARMONIZED"
    assert len(FakeImage.reduce_region_calls) == 12
    for kwargs in FakeImage.reduce_region_calls:
        assert kwargs["scale"] == ndvi.DEFAULT_REDUCE_REGION_SCALE
        assert kwargs["crs"] == ndvi.DEFAULT_REDUCE_REGION_CRS
        assert kwargs["bestEffort"] is True


def test_ndvi_monthly_handles_empty_month(monkeypatch):
    FakeImage.reduce_region_calls = []

    def fake_image_collection(name):
        coll = FakeImageCollection(name)
        coll.empty_months = {5}
        return coll

    fake_ee = SimpleNamespace(
        ImageCollection=fake_image_collection,
        Geometry=lambda geom: geom,
        Filter=SimpleNamespace(calendarRange=lambda start, end, unit: start),
        List=SimpleNamespace(sequence=lambda start, end: FakeSequence(start, end)),
        Reducer=SimpleNamespace(mean=lambda: "mean"),
    )

    monkeypatch.setattr(routes, "ee", fake_ee)
    monkeypatch.setattr(routes, "init_ee", lambda: None)

    request = routes.NDVIRequest(
        geometry={"type": "Point", "coordinates": [0, 0]},
        start="2023-01-01",
        end="2023-12-31",
    )

    data = routes.ndvi_monthly(request)

    assert len(data["data"]) == 12
    missing = next(item for item in data["data"] if item["month"] == 5)
    assert missing["ndvi"] is None
    assert len(FakeImage.reduce_region_calls) == 11


def test_compute_monthly_ndvi_uses_consistent_sampling(monkeypatch):
    FakeImage.reduce_region_calls = []

    def fake_image_collection(name):
        return FakeImageCollection(name)

    fake_ee = SimpleNamespace(
        ImageCollection=fake_image_collection,
        Geometry=lambda geom: geom,
        Filter=SimpleNamespace(calendarRange=lambda start, end, unit: start),
        List=SimpleNamespace(sequence=lambda start, end: FakeSequence(start, end)),
        Reducer=SimpleNamespace(mean=lambda: "mean"),
    )

    monkeypatch.setattr(ndvi, "ee", fake_ee)
    monkeypatch.setattr(ndvi, "init_ee", lambda: None)

    result = ndvi.compute_monthly_ndvi(
        geometry={"type": "Point", "coordinates": [0, 0]},
        year=2023,
    )

    assert len(result) == 12
    assert len(FakeImage.reduce_region_calls) == 12
    for kwargs in FakeImage.reduce_region_calls:
        assert kwargs["scale"] == ndvi.DEFAULT_REDUCE_REGION_SCALE
        assert kwargs["crs"] == ndvi.DEFAULT_REDUCE_REGION_CRS
        assert kwargs["bestEffort"] is True


def test_compute_monthly_ndvi_handles_empty_month(monkeypatch):
    FakeImage.reduce_region_calls = []

    def fake_image_collection(name):
        coll = FakeImageCollection(name)
        coll.empty_months = {7}
        return coll

    fake_ee = SimpleNamespace(
        ImageCollection=fake_image_collection,
        Geometry=lambda geom: geom,
        Filter=SimpleNamespace(calendarRange=lambda start, end, unit: start),
        List=SimpleNamespace(sequence=lambda start, end: FakeSequence(start, end)),
        Reducer=SimpleNamespace(mean=lambda: "mean"),
    )

    monkeypatch.setattr(ndvi, "ee", fake_ee)
    monkeypatch.setattr(ndvi, "init_ee", lambda: None)

    result = ndvi.compute_monthly_ndvi(
        geometry={"type": "Point", "coordinates": [0, 0]},
        year=2023,
    )

    assert len(result) == 12
    missing = next(item for item in result if item["month"] == 7)
    assert missing["ndvi"] is None
    assert len(FakeImage.reduce_region_calls) == 11


def test_cached_ndvi_threads_sampling(monkeypatch):
    call_args = {}

    monkeypatch.setattr(ndvi, "exists", lambda path: False)
    monkeypatch.setattr(ndvi, "upload_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(ndvi, "upload_csv", lambda *args, **kwargs: None)

    def fake_compute(geometry, year, **kwargs):
        call_args["scale"] = kwargs.get("scale")
        call_args["crs"] = kwargs.get("crs")
        return []

    monkeypatch.setattr(ndvi, "compute_monthly_ndvi", fake_compute)

    payload = ndvi.get_or_compute_and_cache_ndvi("abc", {"type": "Point", "coordinates": [0, 0]}, 2024)

    assert payload == {"field_id": "abc", "year": 2024, "data": []}
    assert call_args["scale"] == ndvi.DEFAULT_REDUCE_REGION_SCALE
    assert call_args["crs"] == ndvi.DEFAULT_REDUCE_REGION_CRS
