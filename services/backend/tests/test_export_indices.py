from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pytest
from starlette.datastructures import UploadFile

TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    sys.path.append(str(TEST_DIR))

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

from fake_ee import setup_fake_ee

from app.api import export, s2_indices


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_export_geotiffs_supports_gndvi(monkeypatch):
    context = setup_fake_ee(monkeypatch, export, [0.4, -0.2])

    monkeypatch.setattr(export, "init_ee", lambda: None)
    monkeypatch.setattr(
        export,
        "shapefile_zip_to_geojson",
        lambda _content: {"type": "Point", "coordinates": [0, 0]},
    )

    async def fake_run_in_threadpool(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(export, "run_in_threadpool", fake_run_in_threadpool)

    captured_download: dict[str, str] = {}

    def fake_download_bytes(url: str):
        captured_download["url"] = url
        return b"II*\x00FAKE", "image/tiff"

    monkeypatch.setattr(export, "_download_bytes", fake_download_bytes)

    upload = UploadFile(filename="field.zip", file=io.BytesIO(b"dummy"))

    response = await export.export_geotiffs(
        start_date="2024-01-01",
        end_date="2024-01-31",
        file=upload,
        index="gndvi",
    )

    body = b"".join([chunk async for chunk in response.body_iterator])

    assert (
        response.headers["Content-Disposition"]
        == 'attachment; filename="gndvi_2024-01-01_2024-01-31.zip"'
    )
    assert response.headers["X-Vegetation-Index"] == "gndvi"

    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        assert archive.namelist() == ["gndvi_2024_01.tif"]
        assert archive.read("gndvi_2024_01.tif") == b"II*\x00FAKE"

    assert captured_download["url"] == "https://example.com/download"

    clamp_calls = context["log"].get("clamp_calls", [])
    assert clamp_calls == [(-1.0, 1.0)]

    download_args = context["log"].get("download_args", [])
    assert len(download_args) == 1
    expected_scale = export.resolve_index("gndvi")[0].default_scale
    assert download_args[0]["scale"] == expected_scale


def test_normalise_indices_handles_mixed_case():
    result = export._normalise_indices(
        ["Ndwi_mcfeeters", "NDWI_GAO", "ndsi_soil", "NDWI_GAO"]
    )

    assert result == ["NDWI_McFeeters", "NDWI_Gao", "NDSI_Soil"]


_SIMPLE_POLYGON = {
    "type": "Polygon",
    "coordinates": [
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
        ]
    ],
}


def _build_export_request(indices: list[str]) -> s2_indices.Sentinel2ExportRequest:
    return s2_indices.Sentinel2ExportRequest(
        aoi_geojson=_SIMPLE_POLYGON,
        months=["2024-01"],
        indices=indices,
        export_target="zip",
        aoi_name="Field",
        scale_m=10,
        cloud_prob_max=40,
    )


def test_sentinel2_export_request_accepts_mixed_case_indices():
    request = _build_export_request(
        ["ndwi_mcfeeters", "NdWi_GaO", "NDSI_soil", "NDWI_mcfeeters"]
    )

    assert request.indices == ["NDWI_McFeeters", "NDWI_Gao", "NDSI_Soil"]


def test_start_export_queues_canonical_indices(monkeypatch):
    captured: dict[str, list[str]] = {}

    def fake_create_job(**kwargs):
        captured["index_names"] = kwargs["index_names"]

        class _Job:
            job_id = "job-123"
            state = "pending"

        return _Job()

    monkeypatch.setattr(s2_indices.exports, "create_job", fake_create_job)

    request = _build_export_request(
        ["ndwi_mcfeeters", "NDWI_GAO", "ndsi_soil", "NDWI_mcfeeters"]
    )

    result = s2_indices.start_export(request)

    assert captured["index_names"] == [
        "NDWI_McFeeters",
        "NDWI_Gao",
        "NDSI_Soil",
    ]
    assert result == {"job_id": "job-123", "state": "pending"}
