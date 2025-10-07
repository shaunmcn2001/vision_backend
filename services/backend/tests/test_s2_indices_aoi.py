"""Tests for AOI preparation endpoint used by Sentinel-2 exports."""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import shapefile
from fastapi import HTTPException
from starlette.datastructures import UploadFile

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

from app.api.s2_indices import prepare_aoi_geometry
from app.exports import sanitize_name


@pytest.fixture
def anyio_backend():
    return "asyncio"


_DEF_COORDS = [
    [149.0, -35.0],
    [149.001, -35.0],
    [149.001, -35.001],
    [149.0, -35.001],
    [149.0, -35.0],
]


def _shapefile_zip(coords=_DEF_COORDS) -> bytes:
    with TemporaryDirectory() as tmpdir:
        base = Path(tmpdir) / "field"
        writer = shapefile.Writer(str(base))
        writer.field("name", "C")
        writer.poly([coords])
        writer.record("Field")
        writer.close()

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for suffix in ("shp", "shx", "dbf"):
                archive.write(f"{base}.{suffix}", arcname=f"field.{suffix}")
        return buffer.getvalue()


@pytest.mark.anyio
async def test_prepare_aoi_geometry_returns_geojson(monkeypatch):
    monkeypatch.setenv("MIN_FIELD_HA", "0.1")
    payload = _shapefile_zip()
    upload = UploadFile(filename="field.zip", file=io.BytesIO(payload))

    result = await prepare_aoi_geometry(file=upload, aoi_name="  My Field  ")

    assert result["geometry"]["type"] == "MultiPolygon"
    assert result["area_ha"] > 0
    assert result["aoi_name"] == sanitize_name("  My Field  ")


@pytest.mark.anyio
async def test_prepare_aoi_geometry_enforces_minimum_area(monkeypatch):
    monkeypatch.setenv("MIN_FIELD_HA", "10000")
    payload = _shapefile_zip()
    upload = UploadFile(filename="field.zip", file=io.BytesIO(payload))

    with pytest.raises(HTTPException) as excinfo:
        await prepare_aoi_geometry(file=upload, aoi_name="Tiny")

    assert excinfo.value.status_code == 400
    assert "smaller than minimum" in excinfo.value.detail


@pytest.mark.anyio
async def test_prepare_aoi_geometry_can_skip_area_check(monkeypatch):
    monkeypatch.setenv("MIN_FIELD_HA", "10000")
    payload = _shapefile_zip()
    upload = UploadFile(filename="field.zip", file=io.BytesIO(payload))

    result = await prepare_aoi_geometry(file=upload, enforce_area=False)

    assert result["area_ha"] > 0
