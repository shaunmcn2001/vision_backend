from __future__ import annotations

import io
import shutil
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

TEST_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TEST_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import numpy as np
import pytest
import rasterio
from fastapi import HTTPException
from rasterio.transform import from_origin
import shapefile

from app import exports
from app.api import s2_indices
from app.services import zones


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
        aoi_geojson={
            "type": "Polygon",
            "coordinates": [
                [[0, 0], [0.001, 0], [0.001, 0.001], [0, 0.001], [0, 0]],
            ],
        },
        geometry=None,
    )

    job.zone_state = exports.ZoneExportState(status="ready")
    job.zone_config = exports.ZoneExportConfig(
        n_classes=4,
        cv_mask_threshold=0.3,
        min_mapping_unit_ha=0.5,
        smooth_radius_m=15,
        open_radius_m=10,
        close_radius_m=10,
        simplify_tolerance_m=5,
        simplify_buffer_m=0,
        method="ndvi_kmeans",
        include_stats=True,
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


def _create_zone_artifacts(workdir: Path) -> zones.ZoneArtifacts:
    workdir.mkdir(parents=True, exist_ok=True)

    mean_ndvi_path = workdir / "mean_ndvi.tif"
    with rasterio.open(
        mean_ndvi_path,
        "w",
        driver="GTiff",
        height=1,
        width=1,
        count=1,
        dtype="float32",
        crs=zones.DEFAULT_EXPORT_CRS,
        transform=from_origin(0, 10, zones.DEFAULT_SCALE, zones.DEFAULT_SCALE),
        nodata=-9999.0,
    ) as dst:
        dst.write(np.array([[0.5]], dtype=np.float32), 1)

    raster_path = workdir / "zones.tif"
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=1,
        width=1,
        count=1,
        dtype="uint8",
        crs=zones.DEFAULT_EXPORT_CRS,
        transform=from_origin(0, 10, zones.DEFAULT_SCALE, zones.DEFAULT_SCALE),
        nodata=0,
    ) as dst:
        dst.write(np.array([[1]], dtype="uint8"), 1)

    vector_dir = workdir / "vectors"
    vector_dir.mkdir(exist_ok=True)
    shp_base = vector_dir / "zones"
    with shapefile.Writer(str(shp_base), shapeType=shapefile.POLYGON) as writer:
        writer.autoBalance = 1
        writer.field("zone", "N", decimal=0)
        writer.field("area_ha", "F", decimal=4)
        writer.record(1, 0.1)
        writer.shape(
            {
                "type": "Polygon",
                "coordinates": [[[0, 0], [0, 10], [10, 10], [10, 0], [0, 0]]],
            }
        )
    shp_base.with_suffix(".cpg").write_text("UTF-8")
    shp_base.with_suffix(".prj").write_text("")
    shp_path = shp_base.with_suffix(".shp")
    vector_components = {}
    for ext in ["shp", "dbf", "shx", "prj", "cpg"]:
        member = shp_base.with_suffix(f".{ext}")
        if member.exists():
            vector_components[ext] = str(member)

    stats_path = workdir / "zones_stats.csv"
    stats_path.write_text(
        "zone,area_ha,mean_ndvi,min_ndvi,max_ndvi,pixel_count\n1,0.1,0.5,0.4,0.6,10\n"
    )

    return zones.ZoneArtifacts(
        raster_path=str(raster_path),
        mean_ndvi_path=str(mean_ndvi_path),
        vector_path=str(shp_path),
        vector_components=vector_components,
        zonal_stats_path=str(stats_path),
        working_dir=str(workdir),
    )


def test_create_job_generates_all_exports(monkeypatch):
    monkeypatch.setattr(exports.gee, "initialize", lambda: None)
    monkeypatch.setattr(exports.gee, "geometry_from_geojson", lambda geojson: geojson)

    job = exports.create_job(
        aoi_geojson={
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
        },
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
        aoi_geojson={
            "type": "Polygon",
            "coordinates": [
                [[0, 0], [0.001, 0], [0.001, 0.001], [0, 0.001], [0, 0]],
            ],
        },
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
        aoi_geojson={
            "type": "Polygon",
            "coordinates": [
                [[0, 0], [0.001, 0], [0.001, 0.001], [0, 0.001], [0, 0]],
            ],
        },
        geometry=None,
    )

    default_temp = job.temp_dir
    if default_temp and default_temp.exists():
        shutil.rmtree(default_temp, ignore_errors=True)
    job.temp_dir = None

    job.state = "completed"
    job.updated_at = (
        datetime.utcnow() - exports.JOB_RETENTION_TTL - timedelta(seconds=1)
    )

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
        aoi_geojson={
            "type": "Polygon",
            "coordinates": [
                [[0, 0], [0.001, 0], [0.001, 0.001], [0, 0.001], [0, 0]],
            ],
        },
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

    monkeypatch.setattr(
        exports, "_download_bytes", lambda _url: (b"II*\x00FAKE", "image/tiff")
    )
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
        aoi_geojson={
            "type": "Polygon",
            "coordinates": [
                [[0, 0], [0.001, 0], [0.001, 0.001], [0, 0.001], [0, 0]],
            ],
        },
        geometry={"type": "Point", "coordinates": [0, 0]},
    )

    item = exports.ExportItem(
        month="2024-01",
        index="NDVI",
        variant="analysis",
        file_name="analysis/ndvi_raw.tif",
        image=_FakeImage(),
    )

    monkeypatch.setattr(
        exports, "_download_bytes", lambda _url: (b"II*\x00FAKE", "image/tiff")
    )
    monkeypatch.setattr(exports, "_extract_tiff", lambda payload, _ctype: payload)

    path = exports._download_index_to_path(item, job, tmp_path)

    assert path.exists()
    params = captured["params"]
    assert params["noDataValue"] == -9999
    assert params["formatOptions"] == {"cloudOptimized": False, "noDataValue": -9999}


def test_process_zip_exports_includes_zone_shapefile(tmp_path, monkeypatch):
    job, *_ = _build_zip_job(tmp_path)
    job.zone_state.prefix = exports._zone_prefix(job)

    workdir = tmp_path / "zone_artifacts"
    job.zone_artifacts = _create_zone_artifacts(workdir)

    files, paths = exports._download_zone_artifacts(job, tmp_path)

    file_names = {name for _, name in files}
    assert f"{job.zone_state.prefix}.tif" in file_names
    assert f"{job.zone_state.prefix}_mean_ndvi.tif" in file_names
    assert paths["vectors"].endswith(".shp")
    assert set(paths["vector_components"]) >= {"shp", "dbf", "shx", "prj"}
    if paths["zonal_stats"]:
        assert paths["zonal_stats"].endswith("_zonal_stats.csv")
    if paths["vectors_zip"]:
        assert paths["vectors_zip"] in file_names
    assert paths["geojson"].endswith(".geojson")
    assert paths["mean_ndvi"].endswith("_mean_ndvi.tif")


def test_start_zone_cloud_exports_uploads_zone_files(tmp_path, monkeypatch):
    job, *_ = _build_zip_job(tmp_path)
    job.export_target = "gcs"
    job.zone_state.status = "ready"
    job.zone_state.prefix = exports._zone_prefix(job)

    artifacts_dir = tmp_path / "zone_artifacts"
    job.zone_artifacts = _create_zone_artifacts(artifacts_dir)

    uploaded: dict[str, bytes] = {}

    class FakeBlob:
        def __init__(self, name: str):
            self.name = name

        def upload_from_filename(self, filename: str) -> None:
            uploaded[self.name] = Path(filename).read_bytes()

        def generate_signed_url(self, **_kwargs) -> str:
            return f"https://signed/{self.name}"

    class FakeBucket:
        def __init__(self, name: str):
            self.name = name

        def blob(self, name: str) -> FakeBlob:
            return FakeBlob(name)

    class FakeClient:
        def __init__(self, name: str):
            self._bucket = FakeBucket(name)

        def bucket(self, name: str) -> FakeBucket:
            assert name == self._bucket.name
            return self._bucket

    monkeypatch.setenv("GEE_GCS_BUCKET", "test-bucket")
    fake_client = FakeClient("test-bucket")
    monkeypatch.setattr(exports, "_storage_client", lambda: fake_client)

    exports._start_zone_cloud_exports(job)

    prefix = job.zone_state.prefix
    expected_names = {
        f"{prefix}.tif",
        f"{prefix}_mean_ndvi.tif",
        f"{prefix}.shp",
        f"{prefix}.dbf",
        f"{prefix}.shx",
        f"{prefix}.prj",
        f"{prefix}.cpg",
        f"{prefix}.geojson",
        f"{prefix}_zonal_stats.csv",
        f"{prefix}_shp.zip",
    }

    assert expected_names.issubset(set(uploaded))

    raster_bytes = uploaded[f"{prefix}.tif"]
    assert len(raster_bytes) > 0

    geojson_data = uploaded[f"{prefix}.geojson"].decode()
    assert "FeatureCollection" in geojson_data

    stats_data = uploaded[f"{prefix}_zonal_stats.csv"].decode()
    assert "zone,area_ha" in stats_data

    with zipfile.ZipFile(io.BytesIO(uploaded[f"{prefix}_shp.zip"])) as shp_zip:
        zip_members = set(shp_zip.namelist())
        assert {
            f"{Path(prefix).name}.{ext}" for ext in ["shp", "dbf", "shx", "prj"]
        } <= zip_members

    paths = job.zone_state.paths
    assert paths["raster"] == f"gs://test-bucket/{prefix}.tif"
    assert paths["mean_ndvi"] == f"gs://test-bucket/{prefix}_mean_ndvi.tif"
    assert paths["geojson"] == f"gs://test-bucket/{prefix}.geojson"
    assert paths["vectors_zip"] == f"gs://test-bucket/{prefix}_shp.zip"
    assert (
        job.zone_state.tasks["raster"]["signed_url"] == f"https://signed/{prefix}.tif"
    )
    assert (
        job.zone_state.tasks["mean_ndvi"]["signed_url"]
        == f"https://signed/{prefix}_mean_ndvi.tif"
    )
    assert job.zone_state.status == "completed"
    assert job.zone_artifacts is None
    assert not Path(artifacts_dir).exists()


def test_zone_artifacts_use_raw_geojson_for_mmu(tmp_path, monkeypatch):
    job, *_ = _build_zip_job(tmp_path)
    geometry_sentinel = object()
    job.geometry = geometry_sentinel

    captured: dict[str, object] = {}

    def _write_artifacts(workdir: Path) -> zones.ZoneArtifacts:
        workdir.mkdir(parents=True, exist_ok=True)
        mean_ndvi_path = workdir / "mean_ndvi.tif"
        with rasterio.open(
            mean_ndvi_path,
            "w",
            driver="GTiff",
            height=1,
            width=1,
            count=1,
            dtype="float32",
            crs=zones.DEFAULT_EXPORT_CRS,
            transform=from_origin(0, 10, zones.DEFAULT_SCALE, zones.DEFAULT_SCALE),
            nodata=-9999.0,
        ) as dst:
            dst.write(np.array([[0.5]], dtype=np.float32), 1)

        raster_path = workdir / "zones.tif"
        with rasterio.open(
            raster_path,
            "w",
            driver="GTiff",
            height=1,
            width=1,
            count=1,
            dtype="uint8",
            crs=zones.DEFAULT_EXPORT_CRS,
            transform=from_origin(0, 10, zones.DEFAULT_SCALE, zones.DEFAULT_SCALE),
            nodata=0,
        ) as dst:
            dst.write(np.array([[1]], dtype="uint8"), 1)

        shp_base = workdir / "zones"
        with shapefile.Writer(str(shp_base), shapeType=shapefile.POLYGON) as writer:
            writer.autoBalance = 1
            writer.field("zone", "N", decimal=0)
            writer.field("area_ha", "F", decimal=4)
            writer.record(1, 0.1)
            writer.shape(
                {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [0, 10], [10, 10], [10, 0], [0, 0]]],
                }
            )
        shp_base.with_suffix(".cpg").write_text("UTF-8")
        shp_base.with_suffix(".prj").write_text("")
        shp_path = shp_base.with_suffix(".shp")
        vector_components = {}
        for ext in ["shp", "dbf", "shx", "prj", "cpg"]:
            member = shp_base.with_suffix(f".{ext}")
            if member.exists():
                vector_components[ext] = str(member)

        stats_path = workdir / "zones_stats.csv"
        stats_path.write_text(
            "zone,area_ha,mean_ndvi,min_ndvi,max_ndvi,pixel_count\n1,0.1,0.5,0.4,0.6,10\n"
        )

        return zones.ZoneArtifacts(
            raster_path=str(raster_path),
            mean_ndvi_path=str(mean_ndvi_path),
            vector_path=str(shp_path),
            vector_components=vector_components,
            zonal_stats_path=str(stats_path),
            working_dir=str(workdir),
        )

    def fake_export_selected_period_zones(
        aoi_geojson_or_geom,
        *,
        months,
        aoi_name,
        destination,
        geometry=None,
        min_mapping_unit_ha,
        include_stats=True,
        simplify_tolerance_m=5,
        method=zones.DEFAULT_METHOD,
        **kwargs,
    ):
        captured["aoi_geojson"] = aoi_geojson_or_geom
        captured["geometry"] = geometry
        captured["destination"] = destination
        captured["min_mapping_unit_ha"] = min_mapping_unit_ha
        captured["include_stats"] = include_stats
        captured["simplify_tolerance_m"] = simplify_tolerance_m
        captured["method"] = method
        artifacts = _write_artifacts(tmp_path / "mmu_artifacts")
        return {
            "artifacts": artifacts,
            "metadata": {
                "used_months": list(months),
                "skipped_months": [],
                "min_mapping_unit_applied": False,
                "mmu_applied": False,
            },
            "prefix": "zones/PROD_202401_202401_tiny_field_zones",
        }

    fake_zone_service = SimpleNamespace(
        export_selected_period_zones=fake_export_selected_period_zones,
        export_prefix=lambda name, months: "zones/PROD_202401_202401_tiny_field_zones",
    )

    monkeypatch.setattr(exports, "_zone_service", lambda: fake_zone_service)

    output_dir = tmp_path / "output"
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))

    exports._build_zone_artifacts_for_job(job)

    assert job.zone_state is not None
    assert job.zone_state.metadata.get("min_mapping_unit_applied") is False
    assert job.zone_state.metadata.get("mmu_applied") is False
    assert captured["aoi_geojson"] is job.aoi_geojson
    assert captured["geometry"] is geometry_sentinel
    assert captured["destination"] == "zip"
    assert captured["min_mapping_unit_ha"] == job.zone_config.min_mapping_unit_ha

    exports._process_zip_exports(job)

    assert job.zone_state.paths["raster"].endswith(".tif")
    assert job.zone_state.paths["vectors"].endswith(".shp")
    assert job.zone_state.paths["zonal_stats"].endswith("_zonal_stats.csv")
    assert job.zone_state.paths["geojson"].endswith(".geojson")
    assert job.zone_state.paths["vectors_zip"].endswith("_shp.zip")

    assert job.zip_path is not None and job.zip_path.exists()
    with zipfile.ZipFile(job.zip_path) as archive:
        names = set(archive.namelist())
        assert "zones/PROD_202401_202401_tiny_field_zones.tif" in names
        assert "zones/PROD_202401_202401_tiny_field_zones.shp" in names
        assert "zones/PROD_202401_202401_tiny_field_zones_zonal_stats.csv" in names
        assert any(name.endswith(".geojson") for name in names)
        assert any(name.endswith("_shp.zip") for name in names)

    archive_entries = job.zone_state.metadata.get(exports.ZONE_ARCHIVE_METADATA_KEY)
    assert archive_entries
    assert all(
        entry.get("included_in_zip")
        for entry in archive_entries
        if isinstance(entry, dict)
    )


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
        aoi_geojson={
            "type": "Polygon",
            "coordinates": [
                [[0, 0], [0.001, 0], [0.001, 0.001], [0, 0.001], [0, 0]],
            ],
        },
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

    monkeypatch.setattr(
        exports.gee, "monthly_sentinel2_collection", fake_monthly_collection
    )
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
        aoi_geojson={
            "type": "Polygon",
            "coordinates": [
                [[0, 0], [0.001, 0], [0.001, 0.001], [0, 0.001], [0, 0]],
            ],
        },
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
        {
            "path": str(zone_file),
            "arcname": "zones/PROD_test_zones.tif",
            "included_in_zip": True,
        }
    ]

    exports.JOB_REGISTRY[job.job_id] = job
    zip_path = job.zip_path
    exports.cleanup_job_files(job)

    assert not zone_file.exists()
    assert zip_path is not None and not zip_path.exists()
    assert job.cleaned is True
