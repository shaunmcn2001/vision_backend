"""Earth Engine initialisation helpers."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Mapping, Sequence

import ee
from google.oauth2 import service_account

from app.config import get_settings

_EE_INIT_LOCK = threading.Lock()
_EE_INITIALISED = False


class EarthEngineCredentialsError(RuntimeError):
    """Raised when Earth Engine credentials cannot be loaded."""


def _load_service_account_credentials() -> service_account.Credentials:
    settings = get_settings()
    credentials_path = settings.google_credentials_path
    if not credentials_path:
        raise EarthEngineCredentialsError(
            "GOOGLE_APPLICATION_CREDENTIALS is not configured."
        )

    path = Path(credentials_path)
    if not path.exists():
        raise EarthEngineCredentialsError(
            f"Earth Engine credential file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as handle:
        info = json.load(handle)
    client_email = info.get("client_email")
    if not client_email:
        raise EarthEngineCredentialsError("Service account key missing client_email.")

    scopes = [
        "https://www.googleapis.com/auth/earthengine",
        "https://www.googleapis.com/auth/devstorage.full_control",
    ]
    return service_account.Credentials.from_service_account_file(
        str(path), scopes=scopes
    )


def ensure_ee() -> None:
    """Initialise the Earth Engine client exactly once."""
    global _EE_INITIALISED
    if _EE_INITIALISED:
        return

    with _EE_INIT_LOCK:
        if _EE_INITIALISED:
            return

        settings = get_settings()
        try:
            credentials = _load_service_account_credentials()
            ee.Initialize(credentials, project=settings.gcp_project)
        except EarthEngineCredentialsError:
            ee.Initialize(project=settings.gcp_project)
        _EE_INITIALISED = True


def to_ee_geometry(geometry: Mapping[str, Any]) -> ee.Geometry:
    """Convert GeoJSON-like mapping into an ee.Geometry."""
    ensure_ee()
    try:
        geo_type = geometry.get("type")
        if geo_type == "FeatureCollection":
            features = geometry.get("features")
            if not isinstance(features, Sequence) or not features:
                raise ValueError("FeatureCollection must contain features")
            return ee.FeatureCollection(geometry).geometry()
        if geo_type == "Feature":
            inner = geometry.get("geometry")
            if not isinstance(inner, Mapping):
                raise ValueError("Feature is missing geometry")
            return ee.Geometry(inner)
        return ee.Geometry(geometry)
    except (ee.EEException, TypeError, ValueError, KeyError) as exc:  # pragma: no cover - direct EE exception
        raise ValueError("Invalid AOI geometry") from exc


# Backwards compatibility for older imports
to_geometry = to_ee_geometry
