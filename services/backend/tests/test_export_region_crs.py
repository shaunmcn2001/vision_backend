import asyncio
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

from app.api import export


_ORIGIN_SHIFT = 20037508.342789244


def _to_web_mercator(lon: float, lat: float) -> list[float]:
    clamped_lat = max(min(lat, 89.999999), -89.999999)
    x = lon * _ORIGIN_SHIFT / 180.0
    rad = math.radians(clamped_lat)
    y = math.log(math.tan(math.pi / 4 + rad / 2)) * _ORIGIN_SHIFT / math.pi
    return [x, y]


def _transform_polygon(coords: list[list[list[float]]]) -> list[list[list[float]]]:
    return [[_to_web_mercator(lon, lat) for lon, lat in ring] for ring in coords]


class FakeGeometry:
    def __init__(self, geojson: dict):
        self._geojson = geojson

    def transform(self, crs: str, maxError: float):  # noqa: N802 - match Earth Engine API
        assert crs == "EPSG:3857"
        projected = {
            "type": self._geojson["type"],
            "coordinates": _transform_polygon(self._geojson["coordinates"]),
        }
        return SimpleNamespace(getInfo=lambda: projected)


class RecordingImage:
    def __init__(self):
        self.last_params: dict | None = None

    def getDownloadURL(self, params: dict):
        self.last_params = params
        return "https://example.com/fake.tif"


class DummyUploadFile:
    def __init__(self, content: bytes):
        self.filename = "region.zip"
        self._content = content

    async def read(self) -> bytes:
        return self._content


def test_export_transforms_region_for_epsg_3857(monkeypatch):
    geometry = {
        "type": "Polygon",
        "coordinates": [
            [
                [-123.15, 49.25],
                [-123.05, 49.25],
                [-123.05, 49.3],
                [-123.15, 49.3],
                [-123.15, 49.25],
            ]
        ],
    }
    expected_region = {
        "type": "Polygon",
        "coordinates": _transform_polygon(geometry["coordinates"]),
    }

    recording_image = RecordingImage()

    monkeypatch.setattr(export, "ee", SimpleNamespace(Geometry=lambda geom: FakeGeometry(geom)))
    monkeypatch.setattr(export, "init_ee", lambda: None)
    monkeypatch.setattr(
        export,
        "shapefile_zip_to_geojson",
        lambda content, source_epsg=None: (geometry, []),
    )
    monkeypatch.setattr(
        export,
        "_ndvi_image_for_range",
        lambda geom, start_iso, end_iso: (object(), recording_image),
    )
    monkeypatch.setattr(export, "_collection_size", lambda collection: 1)
    monkeypatch.setattr(
        export,
        "_download_bytes",
        lambda url: (b"II*\x00dummy", "image/tiff"),
    )

    upload = DummyUploadFile(b"fake shapefile bytes")

    response = asyncio.run(
        export.export_geotiffs(
            start_date="2024-01-01",
            end_date="2024-01-31",
            file=upload,
            source_epsg="EPSG:4326",
        )
    )

    assert response.status_code == 200
    assert recording_image.last_params is not None
    params = recording_image.last_params
    assert params["crs"] == "EPSG:3857"
    assert params["region"]["type"] == "Polygon"

    for expected_ring, actual_ring in zip(
        expected_region["coordinates"], params["region"]["coordinates"]
    ):
        for expected_point, actual_point in zip(expected_ring, actual_ring):
            assert actual_point[0] == pytest.approx(expected_point[0], rel=1e-6)
            assert actual_point[1] == pytest.approx(expected_point[1], rel=1e-6)

    assert "X-Geometry-Warnings" not in response.headers


def test_export_includes_geometry_warnings_header(monkeypatch):
    geometry = {
        "type": "Polygon",
        "coordinates": [
            [
                [-123.15, 49.25],
                [-123.05, 49.25],
                [-123.05, 49.3],
                [-123.15, 49.3],
                [-123.15, 49.25],
            ]
        ],
    }
    recording_image = RecordingImage()

    monkeypatch.setattr(export, "ee", SimpleNamespace(Geometry=lambda geom: FakeGeometry(geom)))
    monkeypatch.setattr(export, "init_ee", lambda: None)
    monkeypatch.setattr(
        export,
        "shapefile_zip_to_geojson",
        lambda content, source_epsg=None: (geometry, ["Assumed EPSG:4326."]),
    )
    monkeypatch.setattr(
        export,
        "_ndvi_image_for_range",
        lambda geom, start_iso, end_iso: (object(), recording_image),
    )
    monkeypatch.setattr(export, "_collection_size", lambda collection: 1)
    monkeypatch.setattr(
        export,
        "_download_bytes",
        lambda url: (b"II*\x00dummy", "image/tiff"),
    )

    upload = DummyUploadFile(b"fake shapefile bytes")

    response = asyncio.run(
        export.export_geotiffs(
            start_date="2024-01-01",
            end_date="2024-01-31",
            file=upload,
            source_epsg="EPSG:4326",
        )
    )

    assert response.headers.get("X-Geometry-Warnings") == "Assumed EPSG:4326."
