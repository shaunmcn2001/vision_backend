"""Export orchestration for Sentinel-2 index GeoTIFFs."""
from __future__ import annotations

import io
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import ee
import requests
from google.cloud import storage

from app import gee, index_visualization, indices

MAX_CONCURRENT_EXPORTS = 4
TASK_POLL_SECONDS = 15

SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_-]+")


@dataclass
class ExportItem:
    month: str
    index: str
    file_name: str
    variant: Literal["analysis", "google_earth"] = "analysis"
    status: str = "pending"
    error: Optional[str] = None
    destination_uri: Optional[str] = None
    signed_url: Optional[str] = None
    local_path: Optional[Path] = None
    cleaned: bool = False
    image: Optional[ee.Image] = None
    task: Optional[ee.batch.Task] = None
    is_visualized: bool = False


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
    geometry: ee.Geometry
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


def create_job(
    aoi_geojson: Dict,
    months: List[str],
    index_names: List[str],
    export_target: str,
    aoi_name: str,
    scale_m: int,
    cloud_prob_max: int,
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

    job = ExportJob(
        job_id=job_id,
        export_target=export_target,
        aoi_name=aoi_name,
        safe_aoi_name=safe_name,
        months=months,
        indices=index_names,
        scale_m=scale_m,
        cloud_prob_max=cloud_prob_max,
        geometry=geometry,
        items=items,
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


def _download_index_to_path(item: ExportItem, job: ExportJob, output_dir: Path) -> Path:
    if item.image is None:
        raise ValueError("Missing image for export item")

    params = {
        "scale": job.scale_m,
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
        filePerBand=False,
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


def _process_zip_exports(job: ExportJob) -> None:
    if job.temp_dir is None:
        job.temp_dir = Path(tempfile.mkdtemp(prefix="s2idx_"))

    temp_dir = job.temp_dir
    if temp_dir is None:  # pragma: no cover - defensive
        raise RuntimeError("Temporary directory unavailable")

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

    successful = [item for item in job.items if item.status == "completed" and item.local_path]
    if successful:
        zip_path = temp_dir / f"{job.safe_aoi_name}_sentinel2_indices.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for item in successful:
                archive.write(item.local_path, arcname=item.file_name)
        job.zip_path = zip_path

    if job.all_completed():
        job.state = "completed"
    elif any(item.status == "completed" for item in job.items):
        job.state = "partial"
    else:
        job.state = "failed"
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

    if job.all_completed():
        job.state = "completed"
    elif any(item.status == "completed" for item in job.items):
        job.state = "partial"
    else:
        job.state = "failed"
    job.touch()


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
