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


def test_mask_sentinel2_scales_and_selects_bands():
    class DummyMask:
        def __init__(self, value: bool):
            self.value = bool(value)

        def And(self, other: "DummyMask") -> "DummyMask":
            return DummyMask(self.value and other.value)

    class DummyScalar:
        def __init__(self, value: int):
            self.value = value

        def lte(self, threshold: int) -> DummyMask:
            return DummyMask(self.value <= threshold)

        def bitwiseAnd(self, mask: int) -> "DummyScalar":
            return DummyScalar(self.value & mask)

        def eq(self, other: int) -> DummyMask:
            return DummyMask(self.value == other)

        def neq(self, other: int) -> DummyMask:
            return DummyMask(self.value != other)

    class DummyImage:
        def __init__(self) -> None:
            self.mask_value: bool | None = None
            self.divide_by: int | None = None
            self.selected_bands: list[str] | None = None

        def select(self, band):
            if isinstance(band, list):
                self.selected_bands = band
                return self
            if band == "cloud_probability":
                return DummyScalar(20)
            if band == "QA60":
                return DummyScalar(0)
            if band == "SCL":
                return DummyScalar(5)
            raise AssertionError(f"Unexpected band request: {band}")

        def updateMask(self, mask: DummyMask) -> "DummyImage":
            self.mask_value = mask.value
            return self

        def divide(self, value: int) -> "DummyImage":
            self.divide_by = value
            return self

    image = DummyImage()

    result = gee._mask_sentinel2(image, cloud_prob_max=30)

    assert result is image
    assert image.mask_value is True
    assert image.divide_by == 10_000
    assert image.selected_bands == list(gee.S2_BANDS)
