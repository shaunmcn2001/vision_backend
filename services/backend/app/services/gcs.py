import json
import os
from datetime import timedelta

from google.cloud import storage


def _bucket_name() -> str:
    name = (
        os.environ.get("GEE_GCS_BUCKET") or os.environ.get("GCS_BUCKET") or ""
    ).strip()
    if not name:
        raise RuntimeError(
            "GEE_GCS_BUCKET or GCS_BUCKET env vars are required for GCS access"
        )
    return name


def _client():
    return storage.Client()


def _bucket():
    return _client().bucket(_bucket_name())


def upload_json(data: dict, path: str, content_type: str = "application/json") -> str:
    bucket = _bucket()
    blob = bucket.blob(path)
    blob.cache_control = "no-cache"
    blob.upload_from_string(json.dumps(data), content_type=content_type)
    return f"gs://{bucket.name}/{path}"


def download_json(path: str) -> dict:
    bucket = _bucket()
    blob = bucket.blob(path)
    text = blob.download_as_text()
    return json.loads(text)


def exists(path: str) -> bool:
    bucket = _bucket()
    return bucket.blob(path).exists()


def list_prefix(prefix: str):
    bucket = _bucket()
    return [b.name for b in bucket.list_blobs(prefix=prefix)]


def sign_url(path: str, expires_minutes: int = 60) -> str:
    bucket = _bucket()
    blob = bucket.blob(path)
    return blob.generate_signed_url(
        expiration=timedelta(minutes=expires_minutes), method="GET"
    )
