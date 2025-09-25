from __future__ import annotations

import shutil
from datetime import datetime, timedelta

import pytest
from fastapi import HTTPException

from app import exports
from app.api import s2_indices


@pytest.fixture(autouse=True)
def _clear_job_registry():
    with exports.JOB_LOCK:
        exports.JOB_REGISTRY.clear()
        exports.EVICTED_JOBS.clear()
    yield
    with exports.JOB_LOCK:
        exports.JOB_REGISTRY.clear()
        exports.EVICTED_JOBS.clear()


def _build_zip_job(tmp_path, job_id: str = "job-zip"):
    job = exports.ExportJob(
        job_id=job_id,
        export_target="zip",
        aoi_name="Field",
        safe_aoi_name="field",
        months=["2024-01"],
        indices=["NDVI"],
        scale_m=10,
        cloud_prob_max=40,
        geometry=None,
    )

    default_temp = job.temp_dir
    if default_temp and default_temp.exists():
        shutil.rmtree(default_temp, ignore_errors=True)

    temp_dir = tmp_path / job_id
    temp_dir.mkdir()

    local_path = temp_dir / "ndvi.tif"
    local_path.write_bytes(b"data")

    zip_path = temp_dir / "indices.zip"
    zip_path.write_bytes(b"zip")

    job.items = [
        exports.ExportItem(
            month="2024-01",
            index="NDVI",
            file_name="ndvi.tif",
            status="completed",
            local_path=local_path,
            destination_uri=str(local_path),
        )
    ]
    job.temp_dir = temp_dir
    job.zip_path = zip_path
    job.state = "completed"

    return job, local_path, zip_path, temp_dir


def test_cleanup_job_files_evicts_job_from_registry(tmp_path):
    job, local_path, zip_path, temp_dir = _build_zip_job(tmp_path, "cleanup-job")

    with exports.JOB_LOCK:
        exports.JOB_REGISTRY[job.job_id] = job

    exports.cleanup_job_files(job)

    with exports.JOB_LOCK:
        assert job.job_id not in exports.JOB_REGISTRY

    assert exports.was_job_evicted(job.job_id)
    assert job.cleaned is True
    assert not local_path.exists()
    assert not zip_path.exists()
    assert not temp_dir.exists()


def test_get_job_evicts_expired_jobs(tmp_path):
    job = exports.ExportJob(
        job_id="expired-job",
        export_target="drive",
        aoi_name="Field",
        safe_aoi_name="field",
        months=["2024-01"],
        indices=["NDVI"],
        scale_m=10,
        cloud_prob_max=40,
        geometry=None,
    )

    default_temp = job.temp_dir
    if default_temp and default_temp.exists():
        shutil.rmtree(default_temp, ignore_errors=True)
    job.temp_dir = None

    job.state = "completed"
    job.updated_at = datetime.utcnow() - exports.JOB_RETENTION_TTL - timedelta(seconds=1)

    with exports.JOB_LOCK:
        exports.JOB_REGISTRY[job.job_id] = job

    assert exports.get_job(job.job_id) is None
    assert exports.was_job_evicted(job.job_id)


def test_api_returns_gone_for_evicted_job(tmp_path):
    job, *_ = _build_zip_job(tmp_path, "api-evicted")

    with exports.JOB_LOCK:
        exports.JOB_REGISTRY[job.job_id] = job

    exports.cleanup_job_files(job)

    assert exports.was_job_evicted(job.job_id)

    with pytest.raises(HTTPException) as exc:
        s2_indices.get_status(job.job_id)
    assert exc.value.status_code == 410

    with pytest.raises(HTTPException) as exc:
        s2_indices.download(job.job_id)
    assert exc.value.status_code == 410


def test_download_index_to_path_uses_visualized_format(tmp_path, monkeypatch):
    captured: dict[str, dict[str, object]] = {}

    class _FakeImage:
        def getDownloadURL(self, params: dict[str, object]) -> str:
            captured["params"] = params
            return "https://example.com/download"

    job = exports.ExportJob(
        job_id="job-vis",
        export_target="zip",
        aoi_name="Field",
        safe_aoi_name="field",
        months=["2024-01"],
        indices=["NDVI"],
        scale_m=10,
        cloud_prob_max=40,
        geometry={"type": "Point", "coordinates": [0, 0]},
    )

    item = exports.ExportItem(
        month="2024-01",
        index="NDVI",
        file_name="ndvi.tif",
        image=_FakeImage(),
        is_visualized=True,
    )

    monkeypatch.setattr(exports, "_download_bytes", lambda _url: (b"II*\x00FAKE", "image/tiff"))
    monkeypatch.setattr(exports, "_extract_tiff", lambda payload, _ctype: payload)

    path = exports._download_index_to_path(item, job, tmp_path)

    assert path.exists()
    params = captured["params"]
    assert "noDataValue" not in params
    assert params["formatOptions"] == {"cloudOptimized": False}


def test_download_index_to_path_preserves_nodata_for_scalar(tmp_path, monkeypatch):
    captured: dict[str, dict[str, object]] = {}

    class _FakeImage:
        def getDownloadURL(self, params: dict[str, object]) -> str:
            captured["params"] = params
            return "https://example.com/download"

    job = exports.ExportJob(
        job_id="job-raw",
        export_target="zip",
        aoi_name="Field",
        safe_aoi_name="field",
        months=["2024-01"],
        indices=["NDVI"],
        scale_m=10,
        cloud_prob_max=40,
        geometry={"type": "Point", "coordinates": [0, 0]},
    )

    item = exports.ExportItem(
        month="2024-01",
        index="NDVI",
        file_name="ndvi.tif",
        image=_FakeImage(),
    )

    monkeypatch.setattr(exports, "_download_bytes", lambda _url: (b"II*\x00FAKE", "image/tiff"))
    monkeypatch.setattr(exports, "_extract_tiff", lambda payload, _ctype: payload)

    path = exports._download_index_to_path(item, job, tmp_path)

    assert path.exists()
    params = captured["params"]
    assert params["noDataValue"] == -9999
    assert params["formatOptions"] == {"cloudOptimized": False, "noDataValue": -9999}


def test_run_job_marks_items_visualized(monkeypatch):
    job = exports.ExportJob(
        job_id="job-run",
        export_target="zip",
        aoi_name="Field",
        safe_aoi_name="field",
        months=["2024-01"],
        indices=["NDVI"],
        scale_m=10,
        cloud_prob_max=40,
        geometry={"type": "Point", "coordinates": [0, 0]},
    )
    item = exports.ExportItem(month="2024-01", index="NDVI", file_name="ndvi.tif")
    job.items = [item]

    monkeypatch.setattr(exports.gee, "initialize", lambda: None)

    class _FakeCollection:
        @staticmethod
        def size():
            return type("_Size", (), {"getInfo": staticmethod(lambda: 1)})()

    def fake_monthly_collection(_geometry, _month, _cloud):
        return _FakeCollection(), "composite"

    monkeypatch.setattr(exports.gee, "monthly_sentinel2_collection", fake_monthly_collection)
    monkeypatch.setattr(exports.indices, "compute_index", lambda *_: "index-image")
    monkeypatch.setattr(
        exports.index_visualization,
        "prepare_image_for_export",
        lambda *_: ("visual-image", True),
    )

    def fake_process_zip(job_obj: exports.ExportJob) -> None:
        job_obj.state = "completed"

    monkeypatch.setattr(exports, "_process_zip_exports", fake_process_zip)

    exports._run_job(job)

    assert item.image == "visual-image"
    assert item.is_visualized is True
    assert item.status == "ready"
