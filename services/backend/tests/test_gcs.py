import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

from app.services import gcs


class _FakeBucket:
    def __init__(self, name: str):
        self.name = name


class _FakeClient:
    def __init__(self):
        self.requested_name: str | None = None

    def bucket(self, name: str):
        self.requested_name = name
        return _FakeBucket(name)


def test_bucket_prefers_gee_env(monkeypatch):
    monkeypatch.delenv("GCS_BUCKET", raising=False)
    monkeypatch.setenv("GEE_GCS_BUCKET", "gee-primary")

    fake_client = _FakeClient()
    monkeypatch.setattr(gcs, "_client", lambda: fake_client)

    bucket = gcs._bucket()

    assert isinstance(bucket, _FakeBucket)
    assert bucket.name == "gee-primary"
    assert fake_client.requested_name == "gee-primary"


def test_bucket_requires_configuration(monkeypatch):
    monkeypatch.delenv("GEE_GCS_BUCKET", raising=False)
    monkeypatch.delenv("GCS_BUCKET", raising=False)

    with pytest.raises(RuntimeError) as excinfo:
        gcs._bucket_name()

    assert "GEE_GCS_BUCKET or GCS_BUCKET" in str(excinfo.value)
