from __future__ import annotations

import asyncio
import io
import json
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

from fake_ee import FakeMeanImage, setup_fake_ee

from app.api import export, s2_indices
from app import indices


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
    captured_vis: dict[str, tuple] = {}

    def fake_prepare_image(image, index_name, geometry, scale):
        captured_vis["args"] = (index_name, scale)
        return image, False

    monkeypatch.setattr(
        export.index_visualization,
        "prepare_image_for_export",
        fake_prepare_image,
    )

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
    assert captured_vis["args"] == ("GNDVI", export.resolve_index("gndvi")[0].default_scale)

    clamp_calls = context["log"].get("clamp_calls", [])
    assert clamp_calls == [(-1.0, 1.0)]

    download_args = context["log"].get("download_args", [])
    assert len(download_args) == 1
    expected_scale = export.resolve_index("gndvi")[0].default_scale
    assert download_args[0]["scale"] == expected_scale


@pytest.mark.anyio
async def test_export_geotiffs_uses_visualized_params(monkeypatch):
    context = setup_fake_ee(monkeypatch, export, [0.5])

    monkeypatch.setattr(export, "init_ee", lambda: None)
    monkeypatch.setattr(
        export,
        "shapefile_zip_to_geojson",
        lambda _content: {"type": "Point", "coordinates": [0, 0]},
    )

    async def fake_run_in_threadpool(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(export, "run_in_threadpool", fake_run_in_threadpool)

    def fake_download_bytes(url: str):
        return b"II*\x00FAKE", "image/tiff"

    monkeypatch.setattr(export, "_download_bytes", fake_download_bytes)

    class _VisualizedImage:
        def __init__(self, base):
            self._base = base

        def getDownloadURL(self, params: dict[str, object]) -> str:
            self._base.getDownloadURL(params)
            return "https://example.com/download"

    def fake_prepare_image(image, index_name, geometry, scale):
        return _VisualizedImage(image), True

    monkeypatch.setattr(
        export.index_visualization,
        "prepare_image_for_export",
        fake_prepare_image,
    )

    upload = UploadFile(filename="field.zip", file=io.BytesIO(b"dummy"))

    response = await export.export_geotiffs(
        start_date="2024-01-01",
        end_date="2024-01-31",
        file=upload,
        index="ndvi",
    )

    body = b"".join([chunk async for chunk in response.body_iterator])

    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        assert archive.namelist() == ["ndvi_2024_01.tif"]

    download_args = context["log"].get("download_args", [])
    assert len(download_args) == 1
    params = download_args[0]
    assert "noDataValue" not in params
    assert params["formatOptions"] == {"cloudOptimized": False}


def test_index_image_for_range_returns_clamped_mean(monkeypatch):
    context = setup_fake_ee(monkeypatch, export, [0.4, -0.2])
    definition, parameters = export.resolve_index("ndvi")

    geometry = {"type": "Point", "coordinates": [0, 0]}
    _collection, image = export._index_image_for_range(
        geometry,
        "2024-01-01",
        "2024-01-31",
        definition=definition,
        parameters=parameters,
    )

    assert isinstance(image, FakeMeanImage)
    assert image.value == pytest.approx(0.1)
    assert image.clipped_geom == geometry
    assert image.clamped_to == (-1.0, 1.0)
    assert context["log"].get("reduce_calls") == ["sum", "count", "median", "stdDev"]


def test_finish_coerces_elements_without_clip(monkeypatch):
    class ElementWithoutClip:
        def __init__(self) -> None:
            self.rename_calls = 0
            self.to_float_calls = 0

        def rename(self, name: str):  # pragma: no cover - should not be called
            self.rename_calls += 1
            return self

        def toFloat(self):  # pragma: no cover - should not be called
            self.to_float_calls += 1
            return self

    class WrappedImage:
        def __init__(self, base: ElementWithoutClip) -> None:
            self.base = base
            self.ops: list[tuple] = []

        def rename(self, name: str) -> "WrappedImage":
            self.ops.append(("rename", name))
            return self

        def toFloat(self) -> "WrappedImage":
            self.ops.append(("toFloat", None))
            return self

        def clip(self, geometry):
            self.ops.append(("clip", geometry))
            return self

        def reproject(self, crs: str, _proj, scale: int):
            self.ops.append(("reproject", crs, scale))
            return self

    class FakeEE:
        @staticmethod
        def Image(value):
            return WrappedImage(value)

    monkeypatch.setattr(indices, "ee", FakeEE())

    element = ElementWithoutClip()
    geometry = {"type": "Point", "coordinates": [0, 0]}

    result = indices._finish(element, "TEST", geometry, 30)

    assert isinstance(result, WrappedImage)
    assert result.base is element
    assert element.rename_calls == 0
    assert element.to_float_calls == 0
    assert result.ops == [
        ("rename", "TEST"),
        ("toFloat", None),
        ("clip", geometry),
        ("reproject", "EPSG:4326", 30),
    ]


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


def _build_export_request(
    indices: list[str], production_zones: object | None = None
) -> s2_indices.Sentinel2ExportRequest:
    payload: dict[str, object] = {
        "aoi_geojson": _SIMPLE_POLYGON,
        "months": ["2024-01"],
        "indices": indices,
        "export_target": "zip",
        "aoi_name": "Field",
        "scale_m": 10,
        "cloud_prob_max": 40,
    }
    if production_zones is not None:
        payload["production_zones"] = production_zones
    return s2_indices.Sentinel2ExportRequest(**payload)


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


def test_start_export_includes_zone_config(monkeypatch):
    captured: dict[str, object] = {}

    def fake_create_job(**kwargs):
        captured.update(kwargs)

        class _Job:
            job_id = "job-zone"
            state = "pending"

        return _Job()

    monkeypatch.setattr(s2_indices.exports, "create_job", fake_create_job)

    request = _build_export_request(
        ["NDVI"],
        production_zones={"enabled": True, "n_classes": 4, "cv_mask_threshold": 0.3},
    )

    result = s2_indices.start_export(request)

    zone_config = captured.get("zone_config")
    assert isinstance(zone_config, s2_indices.exports.ZoneExportConfig)
    assert zone_config.n_classes == 4
    assert zone_config.cv_mask_threshold == 0.3
    assert zone_config.min_mapping_unit_ha == s2_indices.zone_service.DEFAULT_MIN_MAPPING_UNIT_HA
    assert result == {"job_id": "job-zone", "state": "pending"}


def test_start_export_enables_zone_config_when_options_provided(monkeypatch):
    captured: dict[str, object] = {}

    def fake_create_job(**kwargs):
        captured.update(kwargs)

        class _Job:
            job_id = "job-zone-options"
            state = "pending"

        return _Job()

    monkeypatch.setattr(s2_indices.exports, "create_job", fake_create_job)

    request = _build_export_request(
        ["NDVI"], production_zones={"n_classes": 5, "mmu_ha": 3.5}
    )

    result = s2_indices.start_export(request)

    assert request.production_zones is not None
    assert request.production_zones.enabled is True
    assert request.production_zones.n_classes == 5
    assert request.production_zones.mmu_ha == 3.5

    zone_config = captured.get("zone_config")
    assert isinstance(zone_config, s2_indices.exports.ZoneExportConfig)
    assert zone_config.n_classes == 5
    assert zone_config.min_mapping_unit_ha == 3.5
    assert result == {"job_id": "job-zone-options", "state": "pending"}


def test_production_zone_boolean_enables_defaults():
    request = _build_export_request(["NDVI"], production_zones=True)
    assert request.production_zones is not None
    assert request.production_zones.enabled is True
    assert request.production_zones.n_classes == s2_indices.zone_service.DEFAULT_N_CLASSES


def test_start_export_returns_server_error_when_gee_initialisation_fails(monkeypatch):
    from app import exports
    from app.main import app

    def _raise_runtime_error():
        raise RuntimeError("Service account JSON missing")

    monkeypatch.setattr(exports.gee, "initialize", _raise_runtime_error)

    payload = {
        "aoi_geojson": _SIMPLE_POLYGON,
        "months": ["2024-01"],
        "indices": ["NDVI"],
        "export_target": "zip",
        "aoi_name": "Field",
        "scale_m": 10,
        "cloud_prob_max": 40,
    }

    async def _post_json(path: str, body: dict[str, object]) -> tuple[int, dict[str, str], bytes]:
        await app.router.startup()
        try:
            raw_body = json.dumps(body).encode("utf-8")
            messages: list[dict[str, object]] = []
            body_sent = False

            async def receive() -> dict[str, object]:
                nonlocal body_sent
                if not body_sent:
                    body_sent = True
                    return {"type": "http.request", "body": raw_body, "more_body": False}
                await asyncio.sleep(0)
                return {"type": "http.disconnect"}

            async def send(message: dict[str, object]) -> None:
                messages.append(message)

            scope = {
                "type": "http",
                "http_version": "1.1",
                "method": "POST",
                "path": path,
                "root_path": "",
                "scheme": "http",
                "client": ("testclient", 50000),
                "server": ("testserver", 80),
                "headers": [
                    (b"host", b"testserver"),
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(raw_body)).encode("ascii")),
                ],
                "query_string": b"",
                "asgi": {"version": "3.0", "spec_version": "2.3"},
            }

            await app(scope, receive, send)
        finally:
            await app.router.shutdown()

        status = 500
        headers: dict[str, str] = {}
        body_bytes = b""
        for message in messages:
            if message.get("type") == "http.response.start":
                status = int(message.get("status", 0))
                headers = {
                    key.decode("latin-1"): value.decode("latin-1")
                    for key, value in message.get("headers", [])
                }
            elif message.get("type") == "http.response.body":
                body_bytes += message.get("body", b"")
        return status, headers, body_bytes

    status, headers, body = asyncio.run(_post_json("/export/s2/indices", payload))

    assert status == 500
    assert headers.get("content-type", "").startswith("application/json")
    detail = json.loads(body.decode("utf-8"))["detail"]
    assert "Earth Engine initialisation failed" in detail
    assert "GEE_SERVICE_ACCOUNT_JSON" in detail
    assert "Service account JSON missing" in detail
