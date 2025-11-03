"""Earth Engine tile session helpers and proxy utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict
from uuid import uuid4

import certifi
import ee
import logging
import requests

from fastapi import HTTPException

from app.config import get_settings
from app.services.earth_engine import ensure_ee


@dataclass
class TileSession:
    token: str
    map_id: str
    ee_token: str
    min_zoom: int
    max_zoom: int
    expires_at: datetime

    @property
    def url_template(self) -> str:
        return f"/api/tiles/{self.token}" + "/{z}/{x}/{y}"


_SESSIONS: Dict[str, TileSession] = {}
logger = logging.getLogger(__name__)


def _ttl() -> timedelta:
    settings = get_settings()
    return timedelta(minutes=settings.tile_session_ttl_minutes)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def cleanup_sessions() -> None:
    """Remove expired tile sessions."""
    now = _now()
    expired = [token for token, session in _SESSIONS.items() if session.expires_at < now]
    for token in expired:
        _SESSIONS.pop(token, None)


def create_tile_session(
    image: ee.Image,
    *,
    vis_params: dict | None = None,
    min_zoom: int = 0,
    max_zoom: int = 22,
) -> dict:
    """Create a managed tile session for the supplied image."""
    ensure_ee()
    cleanup_sessions()
    session_token = uuid4().hex
    vis = vis_params or {}
    map_info = image.getMapId(vis)
    ttl = _ttl()
    session = TileSession(
        token=session_token,
        map_id=map_info["mapid"],
        ee_token=map_info["token"],
        min_zoom=min_zoom,
        max_zoom=max_zoom,
        expires_at=_now() + ttl,
    )
    _SESSIONS[session_token] = session
    return {
        "token": session.token,
        "urlTemplate": session.url_template,
        "minZoom": session.min_zoom,
        "maxZoom": session.max_zoom,
        "expiresAt": session.expires_at.isoformat(),
    }


def resolve_session(token: str) -> TileSession | None:
    cleanup_sessions()
    session = _SESSIONS.get(token)
    if session and session.expires_at > _now():
        return session
    if session:
        _SESSIONS.pop(token, None)
    return None


def fetch_tile_png(session: TileSession, z: int, x: int, y: int) -> bytes:
    """Fetch a tile image from Earth Engine for the given session."""
    url = ee.data.getTileUrl({"mapid": session.map_id, "token": session.ee_token}, x, y, z)
    response = requests.get(url, timeout=20, verify=certifi.where())
    response.raise_for_status()
    return response.content


def fetch_tile_bytes(token: str, z: int, x: int, y: int) -> tuple[bytes, str]:
    """Resolve a session and fetch tile bytes for proxying."""
    session = resolve_session(token)
    if session is None:
        raise HTTPException(status_code=404, detail="Tile session not found or expired.")
    try:
        data = fetch_tile_png(session, z, x, y)
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else 502
        raise HTTPException(status_code=status, detail=f"Earth Engine tile fetch failed: {exc}") from exc
    except requests.RequestException as exc:
        logger.exception("Tile fetch request error for token=%s z=%s x=%s y=%s", token, z, x, y)
        raise HTTPException(status_code=502, detail=f"Tile request failed: {exc}") from exc
    return data, "image/png"
