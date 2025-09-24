from __future__ import annotations

import base64
import json
from types import SimpleNamespace

from app import gee


def _stub_ee(monkeypatch):
    captured: dict[str, object] = {}

    creds_object = object()

    def fake_service_account_credentials(email: str, key_data: str):
        captured["email"] = email
        captured["key_data"] = key_data
        return creds_object

    def fake_initialize(credentials: object) -> None:
        captured["credentials"] = credentials

    monkeypatch.setattr(
        gee,
        "ee",
        SimpleNamespace(
            ServiceAccountCredentials=fake_service_account_credentials,
            Initialize=fake_initialize,
        ),
    )

    monkeypatch.setattr(gee, "_initialized", False, raising=False)

    return captured, creds_object


def test_initialize_accepts_base64_credentials(monkeypatch):
    info = {"client_email": "svc@example.com", "type": "service_account"}
    raw_json = json.dumps(info).encode("utf-8")
    encoded = base64.b64encode(raw_json).decode("ascii")

    monkeypatch.setenv(gee.SERVICE_ACCOUNT_ENV, encoded)
    monkeypatch.delenv(gee.FALLBACK_SERVICE_ACCOUNT_ENV, raising=False)

    captured, creds_object = _stub_ee(monkeypatch)

    gee.initialize(force=True)

    assert captured["email"] == info["client_email"]
    assert json.loads(captured["key_data"]) == info
    assert captured["credentials"] is creds_object


def test_initialize_uses_google_application_credentials_path(tmp_path, monkeypatch):
    info = {"client_email": "svc@example.com", "type": "service_account"}
    cred_file = tmp_path / "service-account.json"
    cred_file.write_text(json.dumps(info), encoding="utf-8")

    monkeypatch.delenv(gee.SERVICE_ACCOUNT_ENV, raising=False)
    monkeypatch.setenv(gee.FALLBACK_SERVICE_ACCOUNT_ENV, str(cred_file))

    captured, creds_object = _stub_ee(monkeypatch)

    gee.initialize(force=True)

    assert captured["email"] == info["client_email"]
    assert json.loads(captured["key_data"]) == info
    assert captured["credentials"] is creds_object
