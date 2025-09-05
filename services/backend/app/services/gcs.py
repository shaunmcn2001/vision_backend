import os, json
from google.cloud import storage

def _client():
    return storage.Client()

def _bucket():
    name = os.environ.get("GCS_BUCKET")
    if not name:
        raise RuntimeError("GCS_BUCKET env var is required")
    return _client().bucket(name)

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
