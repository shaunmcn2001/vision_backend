from __future__ import annotations

import io
import shutil
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

TEST_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TEST_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

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

    local_path = temp_dir / "analysis" / "ndvi_raw.tif"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(b"data")

    zip_path = temp_dir / "indices.zip"
    zip_path.write_bytes(b"zip")

    job.items = [
        exports.ExportItem(
            month="2024-01",
            index="NDVI",
            variant="analysis",
            file_name="analysis/ndvi_raw.tif",
            status="completed",
            local_path=local_path,
            destination_uri=str(local_path),
        )
    ]
    job.temp_dir = temp_dir
    job.zip_path = zip_path
    job.state = "completed"

    return job, local_path, zip_path, temp_dir


def test_create_job_generates_all_exports(monkeypatch):
    monkeypatch.setattr(exports.gee, "initialize", lambda: None)
    monkeypatch.setattr(exports.gee, "geometry_from_geojson", lambda geojson: geojson)

    job = exports.create_job(
        aoi_geojson={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
        months=["2024-01"],
        index_names=["NDVI"],
        export_target="zip",
        aoi_name="Field",
        scale_m=10,
        cloud_prob_max=40,
    )

    assert len(job.items) == 4
    variants = {(item.variant, item.file_name) for item in job.items}
    assert variants == {
        ("analysis", "analysis/NDVI_202401_Field_raw.tif"),
        ("google_earth", "google_earth/NDVI_202401_Field_rgb.tif"),
        ("imagery_true", "imagery/S2_202401_Field_true.tif"),
        ("imagery_false", "imagery/S2_202401_Field_false.tif"),
    }


def test_prepare_imagery_items_sets_visualized(monkeypatch):
    job = exports.ExportJob(
        job_id="job-imagery",
        export_target="zip",
        aoi_name="Field",
        safe_aoi_name="field",
        months=["2024-01"],
        indices=["NDVI"],
        scale_m=10,
        cloud_prob_max=40,
        geometry={"type": "Point", "coordinates": [0, 0]},
    )
    true_item = exports.ExportItem(
        month="2024-01",
        index="S2",
        variant="imagery_true",
        file_name="imagery/S2_202401_field_true.tif",
    )
    false_item = exports.ExportItem(
        month="2024-01",
        index="S2",
        variant="imagery_false",
        file_name="imagery/S2_202401_field_false.tif",
    )
    job.items = [true_item, false_item]

    captured: list[dict[str, object]] = []

    def fake_stretch(
        image,
        bands,
        geometry,
        scale_m,
        min_value=0.0,
        max_value=3000.0,
        gamma=1.0,
    ):
        captured.append(
            {
                "bands": tuple(bands),
                "geometry": geometry,
                "scale": scale_m,
                "min": min_value,
                "max": max_value,
                "gamma": gamma,
            }
        )
        return f"rgb-{bands}"  # pragma: no cover - structure only

    monkeypatch.setattr(exports, "_stretch_to_byte", fake_stretch)

    exports._prepare_imagery_items(job, "2024-01", composite="composite")

    assert true_item.image == "rgb-('B4', 'B3', 'B2')"
    assert true_item.is_visualized is True
    assert true_item.status == "ready"
    assert true_item.scale_override == exports.IMAGERY_SCALE_M

    assert false_item.image == "rgb-('B8', 'B4', 'B3')"
    assert false_item.is_visualized is True
    assert false_item.status == "ready"

    assert false_item.scale_override == exports.IMAGERY_SCALE_M

    assert captured == [
        {
            "bands": ("B4", "B3", "B2"),
            "geometry": job.geometry,
            "scale": exports.IMAGERY_SCALE_M,
            "min": 0,
            "max": 3000,
            "gamma": 1.2,
        },
        {
            "bands": ("B8", "B4", "B3"),
            "geometry": job.geometry,
            "scale": exports.IMAGERY_SCALE_M,
            "min": 0,
            "max": 4000,
            "gamma": 1.3,
        },
    ]


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
        variant="google_earth",
        file_name="google_earth/ndvi_rgb.tif",
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
        variant="analysis",
        file_name="analysis/ndvi_raw.tif",
        image=_FakeImage(),
    )

    monkeypatch.setattr(exports, "_download_bytes", lambda _url: (b"II*\x00FAKE", "image/tiff"))
    monkeypatch.setattr(exports, "_extract_tiff", lambda payload, _ctype: payload)

    path = exports._download_index_to_path(item, job, tmp_path)

    assert path.exists()
    params = captured["params"]
    assert params["noDataValue"] == -9999
    assert params["formatOptions"] == {"cloudOptimized": False, "noDataValue": -9999}


def test_process_zip_exports_includes_zone_shapefile(tmp_path, monkeypatch):
    class _FakeDownloadable:
        def __init__(self, url: str):
            self.url = url
            self.calls: list[dict[str, object]] = []

        def getDownloadURL(self, params: dict[str, object]) -> str:
            self.calls.append(params)
            return self.url

    job = exports.ExportJob(
        job_id="job-zones",
        export_target="zip",
        aoi_name="Field",
        safe_aoi_name="field",
        months=["2024-01"],
        indices=["NDVI"],
        scale_m=10,
        cloud_prob_max=40,
        geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
        zone_config=exports.ZoneExportConfig(
            n_classes=4,
            cv_mask_threshold=0.3,
            min_mapping_unit_ha=0.5,
            smooth_kernel_px=1,
            simplify_tolerance_m=5,
            include_stats=True,
        ),
        zone_state=exports.ZoneExportState(status="ready"),
    )

    job.zone_state.prefix = exports._zone_prefix(job)
    job.zone_artifacts = SimpleNamespace(
        zone_image=_FakeDownloadable("https://example.com/raster"),
        zone_vectors=_FakeDownloadable("https://example.com/vector"),
        zonal_stats=_FakeDownloadable("https://example.com/stats"),
        geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
    )
    job.items = []
    job.temp_dir = tmp_path
    output_dir = tmp_path / "exports_out"
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))

    def _make_zip(contents: dict[str, bytes]) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name, data in contents.items():
                archive.writestr(name, data)
        return buffer.getvalue()

    raster_zip = _make_zip({"raster.tif": b"raster-bytes"})
    vector_zip = _make_zip(
        {
            "zones.shp": b"shp-bytes",
            "zones.shx": b"shx-bytes",
            "zones.dbf": b"dbf-bytes",
            "zones.prj": b"prj-bytes",
        }
    )
    stats_csv = b"zone_id,value\n1,0.5\n"

    responses = {
        "https://example.com/raster": (raster_zip, "application/zip"),
        "https://example.com/vector": (vector_zip, "application/zip"),
        "https://example.com/stats": (stats_csv, "text/csv"),
    }

    monkeypatch.setattr(exports, "_download_bytes", lambda url: responses[url])

    exports._process_zip_exports(job)

    prefix = job.zone_state.prefix
    assert prefix is not None
    assert prefix.startswith("zones/PROD_")
    assert job.zone_state.status == "completed"
    assert job.state == "completed"
    assert job.zone_state.paths["vectors"].endswith(".shp")
    assert job.zone_state.paths["zonal_stats"].endswith("_zonal_stats.csv")
    vector_components = job.zone_state.paths["vector_components"]
    assert vector_components["dbf"].endswith(".dbf")
    assert vector_components["shx"].endswith(".shx")
    assert vector_components["prj"].endswith(".prj")
    assert job.zone_artifacts.zone_vectors.calls[0]["fileFormat"] == "SHP"

    shapefile_components = {
        f"{prefix}.shp",
        f"{prefix}.shx",
        f"{prefix}.dbf",
        f"{prefix}.prj",
    }

    for relative in shapefile_components:
        assert (tmp_path / relative).exists()

    assert job.zip_path is not None and job.zip_path.exists()
    assert job.zip_path.parent == output_dir
    with zipfile.ZipFile(job.zip_path) as archive:
        names = set(archive.namelist())
        expected_entries = {
            f"{prefix}.tif",
            *shapefile_components,
            f"{prefix}_zonal_stats.csv",
        }
        for entry in expected_entries:
            assert entry in names
            assert entry.startswith("zones/")
        assert f"{prefix}.tif" in names
        assert f"{prefix}_zonal_stats.csv" in names

    archive_metadata = job.zone_state.metadata[exports.ZONE_ARCHIVE_METADATA_KEY]
    archived_names = {
        entry["arcname"]
        for entry in archive_metadata
        if isinstance(entry, dict) and entry.get("included_in_zip")
    }
    assert archived_names == {
        f"{prefix}.tif",
        *shapefile_components,
        f"{prefix}_zonal_stats.csv",
    }
    for entry in archive_metadata:
        if not isinstance(entry, dict):
            continue
        path = entry.get("path")
        if path:
            assert Path(path).exists()


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
    analysis_item = exports.ExportItem(
        month="2024-01",
        index="NDVI",
        variant="analysis",
        file_name="analysis/ndvi_raw.tif",
    )
    visual_item = exports.ExportItem(
        month="2024-01",
        index="NDVI",
        variant="google_earth",
        file_name="google_earth/ndvi_rgb.tif",
    )
    job.items = [analysis_item, visual_item]

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

    assert analysis_item.image == "index-image"
    assert analysis_item.is_visualized is False
    assert analysis_item.status == "ready"
    assert visual_item.image == "visual-image"
    assert visual_item.is_visualized is True
    assert visual_item.status == "ready"


def test_cleanup_job_files_removes_zone_exports(tmp_path):
    job = exports.ExportJob(
        job_id="job-clean",
        export_target="zip",
        aoi_name="Field",
        safe_aoi_name="field",
        months=["2024-01"],
        indices=["NDVI"],
        scale_m=10,
        cloud_prob_max=40,
        geometry={"type": "Point", "coordinates": [0, 0]},
        zone_state=exports.ZoneExportState(status="completed"),
    )
    job.temp_dir = tmp_path
    job.zip_path = tmp_path / "archive.zip"
    job.zip_path.write_text("zip")
    zone_file = tmp_path / "zones" / "PROD_test_zones.tif"
    zone_file.parent.mkdir(parents=True, exist_ok=True)
    zone_file.write_text("zone")
    job.zone_state.metadata[exports.ZONE_ARCHIVE_METADATA_KEY] = [
        {"path": str(zone_file), "arcname": "zones/PROD_test_zones.tif", "included_in_zip": True}
    ]

    exports.JOB_REGISTRY[job.job_id] = job
    zip_path = job.zip_path
    exports.cleanup_job_files(job)

    assert not zone_file.exists()
    assert zip_path is not None and not zip_path.exists()
    assert job.cleaned is True
