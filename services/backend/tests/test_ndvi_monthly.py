import sys
from pathlib import Path
from types import SimpleNamespace

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

from app.api import routes


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


class FakeImage:
    def __init__(self, month):
        self._month = month

    def select(self, band):
        return self

    def reduceRegion(self, *args, **kwargs):
        return FakeRegionResult(self._month)


class FakeFilteredCollection:
    def __init__(self, month):
        self._month = month

    def mean(self):
        return FakeImage(self._month)


class FakeMappedCollection:
    def __init__(self, parent):
        self._parent = parent

    def filter(self, month):
        return FakeFilteredCollection(month)


class FakeImageCollection:
    def __init__(self, name):
        self.name = name
        self.geom = None
        self.start = None
        self.end = None

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


def test_ndvi_monthly_partial_year(monkeypatch):
    captured_months = []

    class PartialRegionResult:
        def __init__(self, month):
            self._month = month

        def get(self, key):
            if self._month == 4:
                return None
            return FakeValue(self._month / 100)

    class PartialImage:
        def __init__(self, month):
            self._month = month

        def select(self, band):
            return self

        def reduceRegion(self, *args, **kwargs):
            return PartialRegionResult(self._month)

    class PartialFilteredCollection:
        def __init__(self, month):
            self._month = month

        def mean(self):
            return PartialImage(self._month)

    class PartialMappedCollection:
        def filter(self, month):
            captured_months.append(month)
            return PartialFilteredCollection(month)

    class PartialImageCollection:
        def __init__(self, name):
            self.name = name

        def filterBounds(self, geom):
            return self

        def filterDate(self, start, end):
            self.start = start
            self.end = end
            return self

        def map(self, func):
            return PartialMappedCollection()

    fake_ee = SimpleNamespace(
        ImageCollection=PartialImageCollection,
        Geometry=lambda geom: geom,
        Filter=SimpleNamespace(calendarRange=lambda start, end, unit: start),
        Reducer=SimpleNamespace(mean=lambda: "mean"),
    )

    monkeypatch.setattr(routes, "ee", fake_ee)
    monkeypatch.setattr(routes, "init_ee", lambda: None)

    request = routes.NDVIRequest(
        geometry={"type": "Point", "coordinates": [0, 0]},
        start="2023-03-15",
        end="2023-05-20",
    )

    data = routes.ndvi_monthly(request)

    assert data["ok"] is True
    assert [entry["month"] for entry in data["data"]] == [3, 5]
    assert captured_months == [3, 4, 5]
