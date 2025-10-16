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


def test_initialize_accepts_mapping_credentials(monkeypatch):
    info = {"client_email": "svc@example.com", "type": "service_account"}

    monkeypatch.delenv(gee.SERVICE_ACCOUNT_ENV, raising=False)
    monkeypatch.delenv(gee.FALLBACK_SERVICE_ACCOUNT_ENV, raising=False)

    captured, creds_object = _stub_ee(monkeypatch)

    gee.initialize(force=True, credentials=info)

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
        def __init__(self, scl_value: int) -> None:
            self.scl_value = scl_value
            self.mask_value: bool | None = None
            self.divide_by: int | None = None
            self.selected_bands: list[str] | None = None

        def select(self, band):
            if isinstance(band, list):
                self.selected_bands = band
                return self
            if band == "cloud_probability":
                return DummyScalar(20)
            if band == "SCL":
                return DummyScalar(self.scl_value)
            raise AssertionError(f"Unexpected band request: {band}")

        def updateMask(self, mask: DummyMask) -> "DummyImage":
            self.mask_value = mask.value
            return self

        def divide(self, value: int) -> "DummyImage":
            self.divide_by = value
            return self

    clear_image = DummyImage(5)
    cloudy_image = DummyImage(9)

    result_clear = gee._mask_sentinel2(clear_image, cloud_prob_max=30)
    result_cloudy = gee._mask_sentinel2(cloudy_image, cloud_prob_max=30)

    assert result_clear is clear_image
    assert clear_image.mask_value is True
    assert clear_image.divide_by == 10_000
    assert clear_image.selected_bands == list(gee.S2_BANDS)

    assert result_cloudy is cloudy_image
    assert cloudy_image.mask_value is False
