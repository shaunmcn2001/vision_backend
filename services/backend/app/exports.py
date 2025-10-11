"""Export orchestration for Sentinel-2 index GeoTIFFs."""
from __future__ import annotations

import io
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple, TYPE_CHECKING

import ee
import requests
import shapefile
from google.cloud import storage

from app import gee, index_visualization, indices


logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.services.zones import ZoneArtifacts

MAX_CONCURRENT_EXPORTS = 4
TASK_POLL_SECONDS = 15
IMAGERY_SCALE_M = 10

SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_-]+")


def _output_dir() -> Path:
    base = os.getenv("OUTPUT_DIR", "./exports")
    path = Path(base).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


ExportVariant = Literal[
    "analysis",
    "google_earth",
    "imagery_true",
    "imagery_false",
]


@dataclass
class ExportItem:
    month: str
    index: str
    file_name: str
    variant: ExportVariant = "analysis"
    status: str = "pending"
    error: Optional[str] = None
    destination_uri: Optional[str] = None
    signed_url: Optional[str] = None
    local_path: Optional[Path] = None
    cleaned: bool = False
    image: Optional[ee.Image] = None
    task: Optional[ee.batch.Task] = None
    is_visualized: bool = False
    scale_override: Optional[int] = None


@dataclass
class ZoneExportConfig:
    n_classes: int
    cv_mask_threshold: float
    min_mapping_unit_ha: float
    smooth_radius_m: float
    open_radius_m: float
    close_radius_m: float
    simplify_tolerance_m: float
    simplify_buffer_m: float
    include_stats: bool = True
    apply_stability_mask: Optional[bool] = None


@dataclass
class ZoneExportState:
    status: str = "pending"
    error: Optional[str] = None
    prefix: Optional[str] = None
    paths: Dict[str, object] = field(default_factory=dict)
    tasks: Dict[str, Dict[str, Optional[str]]] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class ExportJob:
    job_id: str
    export_target: str
    aoi_name: str
    safe_aoi_name: str
    months: List[str]
    indices: List[str]
    scale_m: int
    cloud_prob_max: int
    aoi_geojson: Dict
    geometry: ee.Geometry
    zone_config: ZoneExportConfig | None = None
    zone_state: ZoneExportState | None = None
    zone_artifacts: "ZoneArtifacts | None" = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    state: str = "pending"
    error: Optional[str] = None
    items: List[ExportItem] = field(default_factory=list)
    zip_path: Optional[Path] = None
    temp_dir: Optional[Path] = field(
        default_factory=lambda: Path(tempfile.mkdtemp(prefix="s2idx_"))
    )
    cleaned: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def touch(self) -> None:
        self.updated_at = datetime.utcnow()

    def to_status_dict(self) -> Dict:
        zone_payload = None
        if self.zone_config or self.zone_state:
            state = self.zone_state or ZoneExportState(status="pending")
            tasks = {name: dict(values) for name, values in state.tasks.items()}
            zone_payload = {
                "status": state.status,
                "error": state.error,
                "prefix": state.prefix,
                "paths": dict(state.paths),
                "tasks": tasks,
                "metadata": dict(state.metadata),
                "config": asdict(self.zone_config) if self.zone_config else None,
            }

        return {
            "job_id": self.job_id,
            "state": self.state,
            "created_at": self.created_at.isoformat() + "Z",
            "updated_at": self.updated_at.isoformat() + "Z",
            "export_target": self.export_target,
            "aoi_name": self.aoi_name,
            "scale_m": self.scale_m,
            "cloud_prob_max": self.cloud_prob_max,
            "error": self.error,
            "cleaned": self.cleaned,
            "items": [
                {
                    "month": item.month,
                    "index": item.index,
                    "variant": item.variant,
                    "file_name": item.file_name,
                    "status": item.status,
                    "destination_uri": item.destination_uri,
                    "signed_url": item.signed_url,
                    "error": item.error,
                    "cleaned": item.cleaned,
                }
                for item in self.items
            ],
            "zone_exports": zone_payload,
        }

    def all_completed(self) -> bool:
        return all(item.status == "completed" for item in self.items)

    def iter_items_for_month(self, month: str) -> Iterable[ExportItem]:
        return (item for item in self.items if item.month == month)


JOB_REGISTRY: Dict[str, ExportJob] = {}
JOB_LOCK = threading.Lock()

TERMINAL_STATES = frozenset({"completed", "failed", "partial"})
JOB_RETENTION_TTL = timedelta(hours=24)
EVICTED_REGISTRY_TTL = timedelta(hours=6)

ZONE_ARCHIVE_METADATA_KEY = "zip_entries"

EVICTED_JOBS: Dict[str, datetime] = {}


def _prune_evicted_locked(now: datetime) -> None:
    expired_ids = [
        job_id
        for job_id, evicted_at in EVICTED_JOBS.items()
        if now - evicted_at > EVICTED_REGISTRY_TTL
    ]
    for job_id in expired_ids:
        EVICTED_JOBS.pop(job_id, None)


def _record_evicted_locked(job_id: str, now: datetime) -> None:
    _prune_evicted_locked(now)
    EVICTED_JOBS[job_id] = now


def _job_is_terminal(job: ExportJob) -> bool:
    return job.cleaned or job.state in TERMINAL_STATES


def _job_expired(job: ExportJob, now: datetime) -> bool:
    if not _job_is_terminal(job):
        return False
    return now - job.updated_at > JOB_RETENTION_TTL


def _evict_expired_jobs(now: Optional[datetime] = None) -> List[ExportJob]:
    reference_time = now or datetime.utcnow()
    expired: List[ExportJob] = []
    with JOB_LOCK:
        _prune_evicted_locked(reference_time)
        for job_id, job in list(JOB_REGISTRY.items()):
            if _job_expired(job, reference_time):
                expired.append(JOB_REGISTRY.pop(job_id))
                _record_evicted_locked(job_id, reference_time)

    for job in expired:
        if job.export_target == "zip" and not job.cleaned:
            cleanup_job_files(job)

    return expired


def remove_job(job: ExportJob) -> None:
    now = datetime.utcnow()
    with JOB_LOCK:
        existing = JOB_REGISTRY.get(job.job_id)
        if existing is job:
            JOB_REGISTRY.pop(job.job_id, None)
            _record_evicted_locked(job.job_id, now)
        else:
            _record_evicted_locked(job.job_id, now)


def was_job_evicted(job_id: str) -> bool:
    now = datetime.utcnow()
    with JOB_LOCK:
        _prune_evicted_locked(now)
        return job_id in EVICTED_JOBS


_STORAGE_CLIENT: Optional[storage.Client] = None


def _storage_client() -> storage.Client:
    global _STORAGE_CLIENT
    if _STORAGE_CLIENT is None:
        _STORAGE_CLIENT = storage.Client()
    return _STORAGE_CLIENT


from concurrent.futures import ThreadPoolExecutor

EXECUTOR = ThreadPoolExecutor(max_workers=2)


def sanitize_name(name: str) -> str:
    cleaned = SAFE_NAME_PATTERN.sub("_", name.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "aoi"


def _zone_service():  # pragma: no cover - thin wrapper
    from app.services import zones as zone_service

    return zone_service


def _zone_prefix(job: ExportJob) -> str:
    zone_service = _zone_service()
    return zone_service.export_prefix(job.aoi_name, job.months)


def _zone_completed(job: ExportJob) -> bool:
    if not job.zone_config:
        return True
    if job.zone_state is None:
        return False
    return job.zone_state.status in {None, "completed", "skipped"}


def _zone_failed(job: ExportJob) -> bool:
    if not job.zone_config or job.zone_state is None:
        return False
    return job.zone_state.status == "failed"


def create_job(
    aoi_geojson: Dict,
    months: List[str],
    index_names: List[str],
    export_target: str,
    aoi_name: str,
    scale_m: int,
    cloud_prob_max: int,
    zone_config: ZoneExportConfig | None = None,
) -> ExportJob:
    gee.initialize()
    geometry = gee.geometry_from_geojson(aoi_geojson)
    safe_name = sanitize_name(aoi_name or "aoi")
    job_id = uuid.uuid4().hex

    items: List[ExportItem] = []
    for month in months:
        month_code = month.replace("-", "")
        for index in index_names:
            analysis_name = f"analysis/{index}_{month_code}_{safe_name}_raw.tif"
            google_name = f"google_earth/{index}_{month_code}_{safe_name}_rgb.tif"
            items.append(
                ExportItem(
                    month=month,
                    index=index,
                    variant="analysis",
                    file_name=analysis_name,
                )
            )
            items.append(
                ExportItem(
                    month=month,
                    index=index,
                    variant="google_earth",
                    file_name=google_name,
                )
            )

        imagery_true = f"imagery/S2_{month_code}_{safe_name}_true.tif"
        imagery_false = f"imagery/S2_{month_code}_{safe_name}_false.tif"
        items.append(
            ExportItem(
                month=month,
                index="S2",
                variant="imagery_true",
                file_name=imagery_true,
            )
        )
        items.append(
            ExportItem(
                month=month,
                index="S2",
                variant="imagery_false",
                file_name=imagery_false,
            )
        )

    job = ExportJob(
        job_id=job_id,
        export_target=export_target,
        aoi_name=aoi_name,
        safe_aoi_name=safe_name,
        months=months,
        indices=index_names,
        scale_m=scale_m,
        cloud_prob_max=cloud_prob_max,
        aoi_geojson=aoi_geojson,
        geometry=geometry,
        items=items,
        zone_config=zone_config,
        zone_state=ZoneExportState(status="pending") if zone_config else None,
    )

    _evict_expired_jobs()

    with JOB_LOCK:
        _prune_evicted_locked(datetime.utcnow())
        JOB_REGISTRY[job_id] = job

    EXECUTOR.submit(_run_job, job)
    return job


def get_job(job_id: str) -> Optional[ExportJob]:
    _evict_expired_jobs()

    removed_job: Optional[ExportJob] = None
    now = datetime.utcnow()

    with JOB_LOCK:
        job = JOB_REGISTRY.get(job_id)
        if job is None:
            _prune_evicted_locked(now)
            return None

        if _job_expired(job, now):
            removed_job = JOB_REGISTRY.pop(job_id, None)
            _record_evicted_locked(job_id, now)
            job = None
        else:
            _prune_evicted_locked(now)

    if removed_job and removed_job.export_target == "zip" and not removed_job.cleaned:
        cleanup_job_files(removed_job)

    return job


def job_status(job_id: str) -> Optional[Dict]:
    job = get_job(job_id)
    if job is None:
        return None
    with job.lock:
        return job.to_status_dict()


def _download_bytes(url: str) -> Tuple[bytes, Optional[str]]:
    response = requests.get(url, stream=True, timeout=600)
    response.raise_for_status()
    return response.content, response.headers.get("Content-Type")


def _looks_like_tiff(payload: bytes, content_type: Optional[str]) -> bool:
    if content_type and "tif" in content_type.lower():
        return True
    return payload[:4] in (b"II*\x00", b"MM\x00*")


def _extract_tiff(payload: bytes, content_type: Optional[str]) -> bytes:
    if _looks_like_tiff(payload, content_type):
        return payload

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in archive.namelist():
            if name.lower().endswith(".tif"):
                return archive.read(name)
    raise RuntimeError("Download did not contain a GeoTIFF")


def _extract_shapefile_archive(
    url: str, temp_dir: Path, prefix: str
) -> tuple[List[tuple[Path, str]], str]:
    payload, _ = _download_bytes(url)
    base_path = temp_dir / prefix
    base_path.parent.mkdir(parents=True, exist_ok=True)

    extracted: List[tuple[Path, str]] = []
    shapefile_name: Optional[str] = None

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        if not members:
            raise RuntimeError("Shapefile download was empty")

        for info in members:
            member_path = Path(info.filename)
            name = member_path.name
            if not name or name.startswith("__MACOSX"):
                continue

            suffixes = "".join(part.lower() for part in member_path.suffixes)
            if not suffixes:
                continue

            destination = Path(f"{base_path}{suffixes}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(info))

            arcname = f"{prefix}{suffixes}"
            extracted.append((destination, arcname))

            if member_path.suffix.lower() == ".shp":
                shapefile_name = arcname

    if not extracted:
        raise RuntimeError("Shapefile download did not contain any files")
    if shapefile_name is None:
        raise RuntimeError("Shapefile download did not include a .shp file")

    return extracted, shapefile_name


def _download_index_to_path(item: ExportItem, job: ExportJob, output_dir: Path) -> Path:
    if item.image is None:
        raise ValueError("Missing image for export item")

    params = {
        "scale": item.scale_override or job.scale_m,
        "region": job.geometry,
        "crs": "EPSG:4326",
        "filePerBand": False,
        "format": "GEO_TIFF",
        "fileFormat": "GeoTIFF",
        "maxPixels": gee.MAX_PIXELS,
        "name": Path(item.file_name).stem,
    }

    format_options: Dict[str, object] = {"cloudOptimized": False}
    if not item.is_visualized:
        params["noDataValue"] = -9999
        format_options["noDataValue"] = -9999

    params["formatOptions"] = format_options

    url = item.image.getDownloadURL(params)
    payload, content_type = _download_bytes(url)
    tif_bytes = _extract_tiff(payload, content_type)
    output_path = output_dir / item.file_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tif_bytes)
    return output_path


def _gcs_bucket() -> str:
    bucket = os.getenv("GEE_GCS_BUCKET") or os.getenv("GCS_BUCKET")
    if not bucket:
        raise RuntimeError("GEE_GCS_BUCKET or GCS_BUCKET must be set for GCS exports")
    return bucket


def _generate_signed_gcs_url(bucket: str, path: str, expires_minutes: int = 60) -> Optional[str]:
    try:
        client = _storage_client()
        blob = client.bucket(bucket).blob(path)
        return blob.generate_signed_url(expiration=timedelta(minutes=expires_minutes), method="GET")
    except Exception:
        return None


def _export_name_prefix(job: ExportJob, item: ExportItem) -> str:
    file_path = Path(item.file_name)
    relative = file_path.with_suffix("").as_posix()
    components = [job.safe_aoi_name]
    if relative:
        components.append(relative)
    return "/".join(filter(None, components))


def _start_drive_task(item: ExportItem, job: ExportJob) -> ee.batch.Task:
    description = f"{item.index}_{item.month}_{job.safe_aoi_name}"[:100]
    folder = os.getenv("GEE_DRIVE_FOLDER", "Sentinel2_Indices")
    name_prefix = _export_name_prefix(job, item)
    if item.image is None:
        raise ValueError("Missing image for drive export")
    format_options: Dict[str, object] = {"cloudOptimized": False}
    if not item.is_visualized:
        format_options["noDataValue"] = -9999
    task = ee.batch.Export.image.toDrive(
        image=item.image,
        description=description,
        folder=folder,
        fileNamePrefix=name_prefix,
        region=job.geometry,
        scale=job.scale_m,
        crs="EPSG:4326",
        fileFormat="GeoTIFF",
        maxPixels=gee.MAX_PIXELS,
        formatOptions=format_options,
    )
    task.start()
    return task


def _start_gcs_task(item: ExportItem, job: ExportJob, bucket: str) -> ee.batch.Task:
    name_prefix = _export_name_prefix(job, item)
    description = f"{item.index}_{item.month}_{job.safe_aoi_name}"[:100]
    if item.image is None:
        raise ValueError("Missing image for cloud export")
    format_options: Dict[str, object] = {"cloudOptimized": False}
    if not item.is_visualized:
        format_options["noDataValue"] = -9999
    task = ee.batch.Export.image.toCloudStorage(
        image=item.image,
        description=description,
        bucket=bucket,
        fileNamePrefix=name_prefix,
        region=job.geometry,
        scale=job.scale_m,
        crs="EPSG:4326",
        fileFormat="GeoTIFF",
        maxPixels=gee.MAX_PIXELS,
        filePerBand=False,
        formatOptions=format_options,
    )
    task.start()
    return task


def _build_zone_artifacts_for_job(job: ExportJob) -> None:
    if not job.zone_config or job.zone_state is None:
        return

    zone_service = _zone_service()
    prefix = _zone_prefix(job)

    with job.lock:
        job.zone_state.status = "building"
        job.zone_state.error = None
        job.zone_state.prefix = prefix
        job.touch()

    try:
        result = zone_service.export_selected_period_zones(
            job.aoi_geojson,
            months=job.months,
            aoi_name=job.aoi_name,
            geometry=job.geometry,
            cloud_prob_max=job.cloud_prob_max,
            n_classes=job.zone_config.n_classes,
            cv_mask_threshold=job.zone_config.cv_mask_threshold,
            apply_stability_mask=job.zone_config.apply_stability_mask,
            min_mapping_unit_ha=job.zone_config.min_mapping_unit_ha,
            smooth_radius_m=job.zone_config.smooth_radius_m,
            open_radius_m=job.zone_config.open_radius_m,
            close_radius_m=job.zone_config.close_radius_m,
            simplify_tolerance_m=job.zone_config.simplify_tolerance_m,
            simplify_buffer_m=job.zone_config.simplify_buffer_m,
            destination="zip",
            include_stats=job.zone_config.include_stats,
        )
        artifacts = result.get("artifacts")
        metadata = result.get("metadata", {}) or {}
        prefix = result.get("prefix") or prefix
    except Exception as exc:
        logger.exception(
            "Failed to build zone artifacts for job %s (AOI %s): %s",
            job.job_id,
            job.aoi_name,
            exc,
        )
        with job.lock:
            job.zone_artifacts = None
            job.zone_state.status = "failed"
            job.zone_state.error = str(exc)
            if not job.error:
                job.error = str(exc)
            job.touch()
        return

    with job.lock:
        job.zone_artifacts = artifacts
        job.zone_state.status = "ready"
        job.zone_state.error = None
        job.zone_state.prefix = prefix
        job.zone_state.metadata = metadata  # type: ignore[attr-defined]
        job.touch()


def _download_zone_artifacts(
    job: ExportJob, temp_dir: Path
) -> tuple[List[tuple[Path, str]], Dict[str, Optional[str]]]:
    if not job.zone_artifacts or not job.zone_config or job.zone_state is None:
        return [], {}

    artifacts = job.zone_artifacts
    prefix = job.zone_state.prefix or _zone_prefix(job)

    mean_ndvi_src = Path(artifacts.mean_ndvi_path)
    if not mean_ndvi_src.exists():
        raise FileNotFoundError(f"Missing mean NDVI raster at {mean_ndvi_src}")
    mean_ndvi_name = f"{prefix}_mean_ndvi.tif"
    mean_ndvi_path = temp_dir / mean_ndvi_name
    mean_ndvi_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(mean_ndvi_src, mean_ndvi_path)

    raster_src = Path(artifacts.raster_path)
    if not raster_src.exists():
        raise FileNotFoundError(f"Missing raster artifact at {raster_src}")
    raster_name = f"{prefix}.tif"
    raster_path = temp_dir / raster_name
    raster_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(raster_src, raster_path)

    vector_components: Dict[str, str] = {}
    shapefile_members: Dict[str, Path] = {}
    vector_files: List[tuple[Path, str]] = []
    base_name = prefix
    for ext, src in artifacts.vector_components.items():
        dest_name = f"{base_name}.{ext}"
        dest_path = temp_dir / dest_name
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(src, (bytes, bytearray)):
            dest_path.write_bytes(bytes(src))
        else:
            src_path = Path(str(src))
            if not src_path.exists():
                continue
            shutil.copy(src_path, dest_path)
        vector_components[ext] = dest_name
        vector_files.append((dest_path, dest_name))
        if ext.lower() in {"shp", "dbf", "shx", "prj"}:
            shapefile_members[ext.lower()] = dest_path

    geojson_name = f"{prefix}.geojson"
    geojson_path = temp_dir / geojson_name
    geojson_path.parent.mkdir(parents=True, exist_ok=True)
    if shapefile_members.get("shp"):
        try:
            reader = shapefile.Reader(str(shapefile_members["shp"]))
            field_names = [field[0] for field in reader.fields[1:]]
            features = []
            for shape_record in reader.iterShapeRecords():
                properties = {
                    field_names[idx]: shape_record.record[idx]
                    for idx in range(len(field_names))
                }
                geometry = shape_record.shape.__geo_interface__
                features.append(
                    {"type": "Feature", "geometry": geometry, "properties": properties}
                )
            geojson_path.write_text(
                json.dumps({"type": "FeatureCollection", "features": features})
            )
        except Exception as exc:  # pragma: no cover - shapefile failure
            logger.warning("Failed to generate GeoJSON from shapefile: %s", exc)
            geojson_path.write_text(json.dumps({"type": "FeatureCollection", "features": []}))
    else:
        geojson_path.write_text(json.dumps({"type": "FeatureCollection", "features": []}))

    shapefile_zip_name = f"{prefix}_shp.zip"
    shapefile_zip_path = temp_dir / shapefile_zip_name
    shapefile_zip_created = False
    if {"shp", "dbf", "shx", "prj"}.issubset(shapefile_members):
        with zipfile.ZipFile(
            shapefile_zip_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as shp_archive:
            for suffix in ("shp", "dbf", "shx", "prj"):
                member_path = shapefile_members[suffix]
                shp_archive.write(member_path, arcname=member_path.name)
        shapefile_zip_created = True

    stats_name: Optional[str] = None
    stats_path: Optional[Path] = None
    if job.zone_config.include_stats and artifacts.zonal_stats_path:
        stats_src_obj = artifacts.zonal_stats_path
        stats_name = f"{prefix}_zonal_stats.csv"
        stats_path = temp_dir / stats_name
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(stats_src_obj, (bytes, bytearray)):
            stats_path.write_bytes(bytes(stats_src_obj))
        else:
            stats_src = Path(str(stats_src_obj))
            if stats_src.exists():
                shutil.copy(stats_src, stats_path)
            else:
                stats_path = None
                stats_name = None

    files: List[tuple[Path, str]] = [
        (mean_ndvi_path, mean_ndvi_name),
        (raster_path, raster_name),
    ]
    files.extend(vector_files)
    files.append((geojson_path, geojson_name))
    if shapefile_zip_created:
        files.append((shapefile_zip_path, shapefile_zip_name))
    if stats_name and stats_path:
        files.append((stats_path, stats_name))

    paths = {
        "raster": raster_name,
        "mean_ndvi": mean_ndvi_name,
        "vectors": vector_components.get("shp"),
        "vector_components": vector_components,
        "zonal_stats": stats_name,
        "geojson": geojson_name,
        "vectors_zip": shapefile_zip_name if shapefile_zip_created else None,
    }
    return files, paths


def _cleanup_zone_workdir(job: ExportJob) -> None:
    artifacts = job.zone_artifacts
    if artifacts is None:
        return
    workdir = getattr(artifacts, "working_dir", None)
    if workdir:
        try:
            shutil.rmtree(workdir, ignore_errors=True)
        except Exception:
            pass
    job.zone_artifacts = None


def _poll_zone_tasks(
    job: ExportJob,
    tasks: Dict[str, Optional[ee.batch.Task]],
    *,
    bucket: Optional[str] = None,
) -> None:
    if job.zone_state is None:
        return

    active = {name: task for name, task in tasks.items() if task is not None}
    if not active:
        job.zone_state.status = "completed"
        job.touch()
        return

    completed_states = {"COMPLETED", "SUCCEEDED", "COMPLETED_WITH_ERRORS"}
    failed_states = {"FAILED", "CANCELLED"}

    while active:
        time.sleep(TASK_POLL_SECONDS)
        for name, task in list(active.items()):
            try:
                status = task.status() if task else {"state": "FAILED"}
            except Exception as exc:  # pragma: no cover - EE failure surface
                status = {"state": "FAILED", "error_message": str(exc)}
            state = status.get("state")
            task_info = job.zone_state.tasks.get(name, {})
            task_info["state"] = state

            if state in completed_states:
                uris = status.get("destination_uris", []) or []
                destination = uris[0] if uris else None
                task_info["destination_uri"] = destination
                if bucket and destination and destination.startswith(f"gs://{bucket}/"):
                    blob_path = destination[len(f"gs://{bucket}/") :]
                    task_info["signed_url"] = _generate_signed_gcs_url(bucket, blob_path)
                elif destination:
                    task_info["signed_url"] = destination
                active.pop(name, None)
            elif state in failed_states:
                task_info["error"] = (
                    status.get("error_message")
                    or status.get("error_details")
                    or str(status)
                )
                active.pop(name, None)
            job.zone_state.tasks[name] = task_info
        job.touch()

    errors = [info.get("error") for info in job.zone_state.tasks.values() if info.get("error")]
    if errors:
        job.zone_state.status = "failed"
        job.zone_state.error = errors[0]
        if not job.error:
            job.error = errors[0]
    else:
        job.zone_state.status = "completed"
    job.touch()


def _start_zone_cloud_exports(job: ExportJob) -> None:
    if not job.zone_config or job.zone_state is None or job.zone_artifacts is None:
        return

    prefix = job.zone_state.prefix or _zone_prefix(job)
    job.zone_state.prefix = prefix

    staging_dir = Path(tempfile.mkdtemp(prefix="zones_stage_"))

    try:
        zone_files, zone_paths = _download_zone_artifacts(job, staging_dir)
        if job.export_target == "gcs":
            bucket = _gcs_bucket()
            client = _storage_client()
            bucket_ref = client.bucket(bucket)
            uploaded_uris: Dict[str, str] = {}
            for path, arcname in zone_files:
                blob = bucket_ref.blob(arcname)
                blob.upload_from_filename(str(path))
                uploaded_uris[arcname] = f"gs://{bucket}/{arcname}"

            def _to_gcs_uri(name: Optional[str]) -> Optional[str]:
                if not name:
                    return None
                return uploaded_uris.get(name) or f"gs://{bucket}/{name}"

            mapped_paths: Dict[str, object] = {}
            for key, value in zone_paths.items():
                if key == "vector_components" and isinstance(value, dict):
                    mapped_paths[key] = {
                        ext: _to_gcs_uri(name) for ext, name in value.items() if name
                    }
                else:
                    mapped_paths[key] = _to_gcs_uri(value) if isinstance(value, str) else value

            if "geojson" not in mapped_paths:
                mapped_paths["geojson"] = _to_gcs_uri(zone_paths.get("geojson"))

            job.zone_state.paths = mapped_paths
            job.zone_state.tasks = {}
            for key in ("raster", "mean_ndvi", "vectors", "geojson", "zonal_stats", "vectors_zip"):
                destination = mapped_paths.get(key)
                if not isinstance(destination, str):
                    continue
                blob_path = destination[len(f"gs://{bucket}/") :]
                signed_url = _generate_signed_gcs_url(bucket, blob_path)
                job.zone_state.tasks[key] = {
                    "id": None,
                    "state": "COMPLETED",
                    "destination_uri": destination,
                    "signed_url": signed_url,
                    "error": None,
                }

            metadata_entries = job.zone_state.metadata.get(ZONE_ARCHIVE_METADATA_KEY)
            if isinstance(metadata_entries, list):
                for entry in metadata_entries:
                    if not isinstance(entry, dict):
                        continue
                    arcname = entry.get("arcname")
                    if arcname and arcname in uploaded_uris:
                        entry["path"] = uploaded_uris[arcname]
                        entry["destination_uri"] = uploaded_uris[arcname]
                        entry["included_in_zip"] = False
                job.zone_state.metadata[ZONE_ARCHIVE_METADATA_KEY] = metadata_entries

            job.zone_state.status = "completed"
            _cleanup_zone_workdir(job)
            job.touch()
        elif job.export_target == "drive":
            folder = os.getenv("GEE_DRIVE_FOLDER", "Sentinel2_Indices") or "Sentinel2_Indices"
            folder = folder.strip().rstrip("/") or "Sentinel2_Indices"
            if not folder.endswith("zones"):
                folder = f"{folder}/zones"
            drive_prefix = prefix.split("/")[-1]
            staged_dir = _output_dir() / "drive_zones" / job.job_id
            staged_dir.mkdir(parents=True, exist_ok=True)
            staged_paths: Dict[str, str] = {}
            for path, arcname in zone_files:
                dest = staged_dir / arcname
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(path, dest)
                staged_paths[arcname] = str(dest)

            def _to_drive_uri(name: Optional[str]) -> Optional[str]:
                if not name:
                    return None
                final_name = Path(name).name
                return f"drive://{folder}/{final_name}"

            mapped_paths: Dict[str, object] = {}
            for key, value in zone_paths.items():
                if key == "vector_components" and isinstance(value, dict):
                    mapped_paths[key] = {
                        ext: _to_drive_uri(name) for ext, name in value.items() if name
                    }
                else:
                    mapped_paths[key] = _to_drive_uri(value) if isinstance(value, str) else value

            mapped_paths["geojson"] = _to_drive_uri(zone_paths.get("geojson"))

            job.zone_state.paths = mapped_paths
            job.zone_state.tasks = {}
            for key in ("raster", "mean_ndvi", "vectors", "geojson", "zonal_stats", "vectors_zip"):
                destination = mapped_paths.get(key)
                if not isinstance(destination, str):
                    continue
                job.zone_state.tasks[key] = {
                    "id": None,
                    "state": "STAGED",
                    "destination_uri": destination,
                    "signed_url": destination,
                    "error": None,
                }

            job.zone_state.metadata.setdefault("drive_staging", staged_paths)
            job.zone_state.status = "completed"
            _cleanup_zone_workdir(job)
            job.touch()
        else:
            job.zone_state.status = "failed"
            job.zone_state.error = f"Unsupported export target for zones: {job.export_target}"
            if not job.error:
                job.error = job.zone_state.error
            job.touch()
    except Exception as exc:
        logger.exception(
            "Zone cloud export failed for job %s (AOI %s, target %s): %s",
            job.job_id,
            job.aoi_name,
            job.export_target,
            exc,
        )
        job.zone_state.status = "failed"
        job.zone_state.error = str(exc)
        if not job.error:
            job.error = str(exc)
        job.touch()
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
def _process_zip_exports(job: ExportJob) -> None:
    if job.temp_dir is None:
        job.temp_dir = Path(tempfile.mkdtemp(prefix="s2idx_"))

    temp_dir = job.temp_dir
    if temp_dir is None:  # pragma: no cover - defensive
        raise RuntimeError("Temporary directory unavailable")

    zone_files: List[tuple[Path, str]] = []
    zone_paths: Dict[str, Optional[str]] = {}
    zone_archive_entries: List[Dict[str, object]] = []

    for item in job.items:
        if item.image is None:
            item.status = "failed"
            item.error = "Missing composite image"
            continue
        try:
            item.status = "downloading"
            job.touch()
            path = _download_index_to_path(item, job, temp_dir)
            item.local_path = path
            item.destination_uri = str(path)
            item.status = "completed"
        except Exception as exc:
            item.status = "failed"
            item.error = str(exc)
            job.error = exc.args[0] if exc.args else str(exc)
        finally:
            job.touch()

    if job.zone_config and job.zone_state is not None:
        if job.zone_state.status == "ready" and job.zone_artifacts is not None:
            job.zone_state.status = "downloading"
            job.zone_state.error = None
            job.touch()
            try:
                zone_files, zone_paths = _download_zone_artifacts(job, temp_dir)
                zone_archive_entries = [
                    {"path": str(path), "arcname": arcname, "included_in_zip": False}
                    for path, arcname in zone_files
                ]
                if zone_archive_entries:
                    job.zone_state.metadata[ZONE_ARCHIVE_METADATA_KEY] = zone_archive_entries
                _cleanup_zone_workdir(job)
            except Exception as exc:
                logger.exception(
                    "Failed to download zone artifacts for job %s (AOI %s): %s",
                    job.job_id,
                    job.aoi_name,
                    exc,
                )
                job.zone_state.status = "failed"
                job.zone_state.error = str(exc)
                if not job.error:
                    job.error = str(exc)
            else:
                job.zone_state.status = "completed"
                job.zone_state.paths = zone_paths
            finally:
                job.touch()
        elif job.zone_state.status not in {"failed", "completed"}:
            job.zone_state.status = "failed"
            job.zone_state.error = job.zone_state.error or "Zone artifacts unavailable"
            if not job.error and job.zone_state.error:
                job.error = job.zone_state.error
            job.touch()

    successful = [item for item in job.items if item.status == "completed" and item.local_path]
    zip_entries: List[tuple[Path, str]] = [
        (item.local_path, item.file_name)
        for item in successful
        if item.local_path is not None
    ]
    if zone_files:
        zip_entries.extend(zone_files)

    if zip_entries:
        output_dir = _output_dir()
        zip_path = output_dir / f"{job.safe_aoi_name}_sentinel2_indices.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path, arcname in zip_entries:
                archive.write(path, arcname=arcname)
        job.zip_path = zip_path
        if job.zone_state is not None:
            entries = job.zone_state.metadata.get(ZONE_ARCHIVE_METADATA_KEY)
            if isinstance(entries, list):
                for entry in entries:
                    if isinstance(entry, dict):
                        entry["included_in_zip"] = True

    if job.zone_state is not None and job.zone_state.paths:
        keep_keys = (
            "raster",
            "mean_ndvi",
            "vectors",
            "zonal_stats",
            "vector_components",
            "geojson",
            "vectors_zip",
        )
        job.zone_state.paths = {
            key: zone_paths.get(key) for key in keep_keys if key in zone_paths
        }

    if job.all_completed() and _zone_completed(job):
        job.state = "completed"
    elif any(item.status == "completed" for item in job.items) or (
        job.zone_state is not None and job.zone_state.status == "completed"
    ):
        job.state = "partial"
    else:
        job.state = "failed"

    if _zone_failed(job) and job.zone_state and job.zone_state.error and not job.error:
        job.error = job.zone_state.error
    job.touch()


def _process_cloud_exports(job: ExportJob) -> None:
    bucket_name = _gcs_bucket() if job.export_target == "gcs" else None
    queue: List[ExportItem] = [item for item in job.items if item.image is not None]
    active: List[ExportItem] = []

    while queue or active:
        while queue and len(active) < MAX_CONCURRENT_EXPORTS:
            item = queue.pop(0)
            try:
                item.status = "exporting"
                job.touch()
                if job.export_target == "drive":
                    item.task = _start_drive_task(item, job)
                else:
                    item.task = _start_gcs_task(item, job, bucket_name)
                active.append(item)
            except Exception as exc:
                item.status = "failed"
                item.error = str(exc)
                job.touch()

        if not active:
            break

        time.sleep(TASK_POLL_SECONDS)

        for item in list(active):
            status = item.task.status() if item.task else {"state": "FAILED"}
            state = status.get("state")
            if state in {"COMPLETED", "SUCCEEDED", "COMPLETED_WITH_ERRORS"}:
                uris = status.get("destination_uris", []) or []
                item.destination_uri = uris[0] if uris else None
                if job.export_target == "gcs" and item.destination_uri:
                    prefix = f"gs://{bucket_name}/"
                    if item.destination_uri.startswith(prefix):
                        blob_path = item.destination_uri[len(prefix) :]
                        item.signed_url = _generate_signed_gcs_url(bucket_name, blob_path)
                elif job.export_target == "drive" and uris:
                    item.signed_url = uris[0]
                item.status = "completed"
                active.remove(item)
                job.touch()
            elif state in {"FAILED", "CANCELLED"}:
                item.status = "failed"
                item.error = status.get("error_message") or status.get("error_details") or str(status)
                active.remove(item)
                job.touch()

    if job.zone_config and job.zone_state is not None:
        if job.zone_state.status == "ready":
            _start_zone_cloud_exports(job)
        elif job.zone_state.status == "building":
            job.zone_state.status = "failed"
            job.zone_state.error = job.zone_state.error or "Zone artifacts unavailable"
            if not job.error and job.zone_state.error:
                job.error = job.zone_state.error
            job.touch()

    if job.all_completed() and _zone_completed(job):
        job.state = "completed"
    elif any(item.status == "completed" for item in job.items) or (
        job.zone_state is not None and job.zone_state.status == "completed"
    ):
        job.state = "partial"
    else:
        job.state = "failed"
    if _zone_failed(job) and job.zone_state and job.zone_state.error and not job.error:
        job.error = job.zone_state.error
    job.touch()


def _stretch_to_byte(
    image: ee.Image,
    bands: Iterable[str],
    geometry: ee.Geometry,
    scale_m: int,
    min_value: float = 0.0,
    max_value: float = 3000.0,
    gamma: float = 1.0,
) -> ee.Image:
    base_image = ee.Image(image)
    span = max(max_value - min_value, 1.0)
    selected = base_image.select(list(bands)).resample("bilinear")
    scaled = selected.subtract(min_value).divide(span).clamp(0, 1)
    if gamma not in (1.0, 1):
        try:
            inv_gamma = 1.0 / gamma
        except ZeroDivisionError:
            inv_gamma = 1.0
        scaled = scaled.pow(ee.Number(inv_gamma))
    byte_image = ee.Image(scaled.multiply(255).toUint8())
    return byte_image.clip(geometry).reproject("EPSG:4326", None, scale_m)


def _prepare_imagery_items(job: ExportJob, month: str, composite: ee.Image) -> None:
    month_items = list(job.iter_items_for_month(month))

    true_items = [item for item in month_items if item.variant == "imagery_true"]
    if true_items:
        try:
            true_rgb = _stretch_to_byte(
                composite,
                ("B4", "B3", "B2"),
                job.geometry,
                IMAGERY_SCALE_M,
                min_value=0,
                max_value=3000,
                gamma=1.2,
            )
        except Exception as exc:
            for item in true_items:
                item.status = "failed"
                item.error = str(exc)
        else:
            for item in true_items:
                item.image = true_rgb
                item.is_visualized = True
                item.scale_override = IMAGERY_SCALE_M
                item.status = "ready"

    false_items = [item for item in month_items if item.variant == "imagery_false"]
    if false_items:
        try:
            false_rgb = _stretch_to_byte(
                composite,
                ("B8", "B4", "B3"),
                job.geometry,
                IMAGERY_SCALE_M,
                min_value=0,
                max_value=4000,
                gamma=1.3,
            )
        except Exception as exc:
            for item in false_items:
                item.status = "failed"
                item.error = str(exc)
        else:
            for item in false_items:
                item.image = false_rgb
                item.is_visualized = True
                item.scale_override = IMAGERY_SCALE_M
                item.status = "ready"


def _run_job(job: ExportJob) -> None:
    with job.lock:
        job.state = "running"
        job.touch()

    try:
        gee.initialize()
    except Exception as exc:
        with job.lock:
            job.state = "failed"
            job.error = f"Earth Engine init failed: {exc}"
            job.touch()
        return

    try:
        for month in job.months:
            collection, composite = gee.monthly_sentinel2_collection(job.geometry, month, job.cloud_prob_max)
            try:
                size = int(collection.size().getInfo())
            except Exception:
                size = 0
            if size == 0:
                for item in job.iter_items_for_month(month):
                    item.status = "failed"
                    item.error = "No imagery available"
                job.touch()
                continue

            month_items = list(job.iter_items_for_month(month))
            _prepare_imagery_items(job, month, composite)
            for index_name in job.indices:
                index_items = [item for item in month_items if item.index == index_name]
                if not index_items:
                    continue
                try:
                    index_image = indices.compute_index(
                        composite, index_name, job.geometry, job.scale_m
                    )
                except Exception as exc:
                    for item in index_items:
                        item.status = "failed"
                        item.error = str(exc)
                    job.touch()
                    continue

                analysis_items = [
                    item for item in index_items if item.variant == "analysis"
                ]
                for item in analysis_items:
                    item.image = index_image
                    item.is_visualized = False
                    item.status = "ready"

                visual_items = [
                    item for item in index_items if item.variant == "google_earth"
                ]
                if visual_items:
                    try:
                        visual_image, is_visualized = index_visualization.prepare_image_for_export(
                            index_image, index_name, job.geometry, job.scale_m
                        )
                    except Exception as exc:
                        for item in visual_items:
                            item.status = "failed"
                            item.error = str(exc)
                    else:
                        if is_visualized:
                            for item in visual_items:
                                item.image = visual_image
                                item.is_visualized = True
                                item.status = "ready"
                        else:
                            for item in visual_items:
                                item.status = "failed"
                                item.error = "Visualisation unavailable"
                job.touch()

        _build_zone_artifacts_for_job(job)

        if job.export_target == "zip":
            _process_zip_exports(job)
        else:
            _process_cloud_exports(job)
    except Exception as exc:
        with job.lock:
            job.state = "failed"
            job.error = str(exc)
            job.touch()


def get_zip_path(job_id: str) -> Path:
    job = get_job(job_id)
    if job is None:
        raise KeyError(job_id)
    if job.export_target != "zip":
        raise ValueError("ZIP download is only available for zip export jobs")
    if job.zip_path is None or not job.zip_path.exists():
        raise FileNotFoundError("ZIP archive not ready")
    return job.zip_path


def cleanup_job_files(job: ExportJob) -> None:
    """Remove temporary files for completed ZIP export jobs."""

    if job.export_target != "zip":
        remove_job(job)
        return

    paths_to_remove: List[Path] = []
    zip_path: Optional[Path] = None
    temp_dir: Optional[Path] = None

    with job.lock:
        temp_dir = job.temp_dir
        zip_path = job.zip_path

        if not job.cleaned:
            paths_to_remove = []
            for item in job.items:
                local_path = item.local_path
                if local_path is not None:
                    paths_to_remove.append(local_path)
                item.local_path = None
                item.destination_uri = None
                item.signed_url = None
                if local_path is not None or item.status == "completed":
                    item.cleaned = True

            if job.zone_state is not None:
                metadata_entries = job.zone_state.metadata.get(ZONE_ARCHIVE_METADATA_KEY)
                if isinstance(metadata_entries, list):
                    for entry in metadata_entries:
                        if not isinstance(entry, dict):
                            continue
                        path_str = entry.get("path")
                        included = entry.get("included_in_zip")
                        if path_str and (included is True or included is None):
                            paths_to_remove.append(Path(path_str))
                        entry["included_in_zip"] = False
                    job.zone_state.metadata[ZONE_ARCHIVE_METADATA_KEY] = metadata_entries

            job.temp_dir = None
            job.zip_path = None
            job.cleaned = True
            job.touch()

        remove_job(job)

    for path in paths_to_remove:
        if path is None:
            continue
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass

    if zip_path and zip_path.exists():
        try:
            zip_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass

    if temp_dir and temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
