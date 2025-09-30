import sys
from pathlib import Path
from types import SimpleNamespace

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

from app.api import routes
from app.services import ndvi as ndvi_service


class FakeValue:
    def __init__(self, val):
        self._val = val

    def getInfo(self):
        return self._val


class FakeRegionResult:
    def __init__(self, month, capture):
        self._month = month
        self._capture = capture

    def get(self, key):
        self._capture.setdefault("reduce_keys", []).append(key)
        return FakeValue(self._month / 100)


class FakeIndexBand:
    def __init__(self, month, capture):
        self._month = month
        self._capture = capture
        self.band_name: str | None = None

    def rename(self, name: str):
        self.band_name = name
        self._capture.setdefault("renamed_bands", []).append(name)
        return self


class FakeImage:
    def __init__(self, month, capture):
        self._month = month
        self._capture = capture

    def normalizedDifference(self, bands):
        self._capture.setdefault("normalized_difference", []).append(tuple(bands))
        return FakeIndexBand(self._month, self._capture)

    def addBands(self, image):
        self._capture.setdefault("added_bands", []).append(getattr(image, "band_name", None))
        return self

    def select(self, band):
        self._capture.setdefault("selected_bands", []).append(band)
        return self

    def reduceRegion(self, *args, **kwargs):
        self._capture.setdefault("reduced_months", []).append(self._month)
        return FakeRegionResult(self._month, self._capture)


class FakeFilteredCollection:
    def __init__(self, month, capture):
        self._month = month
        self._capture = capture

    def mean(self):
        self._capture.setdefault("mean_months", []).append(self._month)
        return FakeImage(self._month, self._capture)


class FakeMappedCollection:
    def __init__(self, func, capture):
        self._func = func
        self._capture = capture

    def filter(self, month):
        self._capture.setdefault("filtered_months", []).append(month)
        self._func(FakeImage(month, self._capture))
        return FakeFilteredCollection(month, self._capture)


class FakeImageCollection:
    def __init__(self, name, capture):
        self.name = name
        self._capture = capture
        self._capture.setdefault("collections", []).append(name)

    def filterBounds(self, geom):
        self._capture["geometry"] = geom
        return self

    def filterDate(self, start, end):
        self._capture["date_range"] = (start, end)
        return self

    def map(self, func):
        self._capture["map_func"] = func
        return FakeMappedCollection(func, self._capture)


def make_fake_ee(capture):
    return SimpleNamespace(
        ImageCollection=lambda name: FakeImageCollection(name, capture),
        Geometry=lambda geom: geom,
        Filter=SimpleNamespace(calendarRange=lambda start, end, unit: start),
        Reducer=SimpleNamespace(mean=lambda: "mean"),
        Image=SimpleNamespace(constant=lambda value: value),
    )


def _patch_ndvi_service(monkeypatch, capture):
    monkeypatch.setattr(ndvi_service, "ee", make_fake_ee(capture))
    monkeypatch.setattr(ndvi_service, "init_ee", lambda: None)


def test_monthly_index_defaults(monkeypatch):
    capture: dict = {}
    _patch_ndvi_service(monkeypatch, capture)

    request = routes.MonthlyIndexRequest(
        geometry={"type": "Point", "coordinates": [0, 0]},
        start="2023-01-01",
        end="2023-12-31",
    )

    response = routes.ndvi_monthly(request)

    assert response["ok"] is True
    assert response["index"] == {
        "code": "ndvi",
        "band": "NDVI",
        "valid_range": [-1.0, 1.0],
        "parameters": {},
    }
    assert len(response["data"]) == 12
    assert response["data"][0] == {"month": 1, "ndvi": 0.01}

    assert capture["collections"] == ["COPERNICUS/S2_SR_HARMONIZED"]
    assert set(capture["normalized_difference"]) == {("B8", "B4")}
    assert set(capture["selected_bands"]) == {"NDVI"}
    assert set(capture["reduce_keys"]) == {"NDVI"}


def test_monthly_index_gndvi(monkeypatch):
    capture: dict = {}
    _patch_ndvi_service(monkeypatch, capture)

    request = routes.MonthlyIndexRequest(
        geometry={"type": "Point", "coordinates": [0, 0]},
        start="2023-01-01",
        end="2023-12-31",
        index=routes.IndexSelection(code="gndvi"),
    )

    response = routes.ndvi_monthly(request)

    assert response["index"]["code"] == "gndvi"
    assert response["index"]["band"] == "GNDVI"
    assert response["data"][0] == {"month": 1, "gndvi": 0.01}
    assert set(capture["normalized_difference"]) == {("B8", "B3")}


def test_monthly_index_ndre_parameters(monkeypatch):
    capture: dict = {}
    _patch_ndvi_service(monkeypatch, capture)

    request = routes.MonthlyIndexRequest(
        geometry={"type": "Point", "coordinates": [0, 0]},
        start="2023-01-01",
        end="2023-12-31",
        index=routes.IndexSelection(code="ndre", parameters={"nir_band": "B8A"}),
    )

    response = routes.ndvi_monthly(request)

    assert response["index"]["code"] == "ndre"
    assert response["index"]["parameters"] == {"nir_band": "B8A", "red_edge_band": "B5"}
    assert set(capture["normalized_difference"]) == {("B8A", "B5")}


def test_cache_paths_and_payload(monkeypatch):
    captured: dict = {}

    monkeypatch.setattr(ndvi_service, "exists", lambda path: False)

    def fake_upload_json(payload, path):
        captured["json"] = (payload, path)

    def fake_upload_csv(rows, path, key):
        captured["csv"] = (rows, path, key)

    def fake_compute(*args, **kwargs):
        captured["compute_args"] = (args, kwargs)
        return {
            "index": {
                "code": "gndvi",
                "band": "GNDVI",
                "valid_range": [-1.0, 1.0],
                "parameters": {},
            },
            "data": [{"month": 1, "gndvi": 0.42}],
        }

    monkeypatch.setattr(ndvi_service, "upload_json", fake_upload_json)
    monkeypatch.setattr(ndvi_service, "upload_index_csv", fake_upload_csv)
    monkeypatch.setattr(ndvi_service, "compute_monthly_index_for_year", fake_compute)

    payload = ndvi_service.get_or_compute_and_cache_index(
        "field-1",
        {"type": "Point", "coordinates": [0, 0]},
        2020,
        index_code="gndvi",
    )

    assert captured["json"][1] == "index-results/gndvi/field-1/2020.json"
    assert captured["csv"][1] == "index-results/gndvi/field-1/2020.csv"
    assert captured["csv"][2] == "gndvi"
    assert payload == {
        "field_id": "field-1",
        "year": 2020,
        "index": {
            "code": "gndvi",
            "band": "GNDVI",
            "valid_range": [-1.0, 1.0],
            "parameters": {},
        },
        "data": [{"month": 1, "gndvi": 0.42}],
    }


def test_ndvi_links_uses_gee_bucket(monkeypatch):
    monkeypatch.delenv("GCS_BUCKET", raising=False)
    monkeypatch.setenv("GEE_GCS_BUCKET", "gee-only")
    monkeypatch.setattr(routes, "sign_url", lambda path: f"signed://{path}")

    response = routes.ndvi_links("field-7", 2023, index="ndvi")

    assert response["json"]["gs"] == "gs://gee-only/index-results/ndvi/field-7/2023.json"
    assert response["json"]["signed"] == "signed://index-results/ndvi/field-7/2023.json"
    assert response["csv"]["gs"] == "gs://gee-only/index-results/ndvi/field-7/2023.csv"
    assert response["csv"]["signed"] == "signed://index-results/ndvi/field-7/2023.csv"
