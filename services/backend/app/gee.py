"""Shared Google Earth Engine helpers for Sentinel-2 processing."""
from __future__ import annotations

import base64
import binascii
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import ee

S2_SR_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
S2_CLOUD_PROB_COLLECTION = "COPERNICUS/S2_CLOUD_PROBABILITY"
S2_BANDS: Iterable[str] = (
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B11",
    "B12",
)
SERVICE_ACCOUNT_ENV = "GEE_SERVICE_ACCOUNT_JSON"
FALLBACK_SERVICE_ACCOUNT_ENV = "GOOGLE_APPLICATION_CREDENTIALS"
MAX_PIXELS = int(1e13)

_initialized = False


def _looks_like_path(value: str) -> bool:
    return any(sep in value for sep in ("/", "\\")) or value.lower().endswith(".json")


def _load_service_account_info(raw_credentials: str) -> Dict:
    trimmed = raw_credentials.strip()
    if not trimmed:
        raise RuntimeError("Service account credential value is empty")

    if trimmed.startswith("{"):
        try:
            info = json.loads(trimmed)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Service account credential JSON string is invalid") from exc
        if not isinstance(info, dict):
            raise RuntimeError("Service account credential JSON must be an object")
        return info

    cred_path = Path(trimmed)
    if cred_path.exists():
        try:
            with cred_path.open("r", encoding="utf-8") as fh:
                info = json.load(fh)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Credential path {cred_path} did not contain valid JSON") from exc
        if not isinstance(info, dict):
            raise RuntimeError(f"Credential path {cred_path} must contain a JSON object")
        return info

    try:
        decoded_bytes = base64.b64decode(trimmed, validate=True)
    except (binascii.Error, ValueError):
        decoded_bytes = None

    if decoded_bytes:
        try:
            decoded = decoded_bytes.decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise RuntimeError("Decoded service account credentials were not valid UTF-8") from exc
        if not decoded:
            raise RuntimeError("Decoded service account credential JSON is empty")
        if not decoded.startswith("{"):
            raise RuntimeError("Decoded service account credentials must be a JSON object")
        try:
            info = json.loads(decoded)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Decoded service account credential JSON is invalid") from exc
        if not isinstance(info, dict):
            raise RuntimeError("Decoded service account credential JSON must describe an object")
        return info

    if _looks_like_path(trimmed):
        raise RuntimeError(f"Credential path {cred_path} does not exist")

    raise RuntimeError(
        "Service account credentials must be provided as JSON, base64-encoded JSON, or a path to a JSON file"
    )


def initialize(force: bool = False) -> None:
    """Initialise Earth Engine using configured service account credentials."""
    global _initialized
    if _initialized and not force:
        return

    primary = (os.getenv(SERVICE_ACCOUNT_ENV) or "").strip()
    fallback = (os.getenv(FALLBACK_SERVICE_ACCOUNT_ENV) or "").strip()
    raw_credentials = primary or fallback
    if not raw_credentials:
        raise RuntimeError(
            "Set GEE_SERVICE_ACCOUNT_JSON or GOOGLE_APPLICATION_CREDENTIALS to a service account credential JSON string, "
            "base64-encoded JSON, or file path."
        )

    info = _load_service_account_info(raw_credentials)

    email = info.get("client_email")
    if not email:
        raise RuntimeError("Service account JSON is missing client_email")

    credentials = ee.ServiceAccountCredentials(email, key_data=json.dumps(info))
    ee.Initialize(credentials)
    _initialized = True


def geometry_from_geojson(aoi_geojson: Dict) -> ee.Geometry:
    """Convert GeoJSON to an Earth Engine geometry."""
    return ee.Geometry(aoi_geojson)


def month_date_range(month: str) -> Tuple[str, str]:
    """Return ISO8601 start (inclusive) and end (exclusive) datetimes for a month string."""
    start = datetime.strptime(month, "%Y-%m")
    if start.month == 12:
        end = start.replace(year=start.year + 1, month=1, day=1)
    else:
        end = start.replace(month=start.month + 1, day=1)
    start_iso = start.strftime("%Y-%m-%d")
    end_iso = end.strftime("%Y-%m-%d")
    return start_iso, end_iso


def _attach_cloud_probability(collection: ee.ImageCollection, probability: ee.ImageCollection) -> ee.ImageCollection:
    join = ee.Join.saveFirst("cloud_prob")
    matches = join.apply(
        primary=collection,
        secondary=probability,
        condition=ee.Filter.equals(leftField="system:index", rightField="system:index"),
    )

    def _add_probability(image: ee.Image) -> ee.Image:
        cloud_match = image.get("cloud_prob")
        probability_band = ee.Image(
            ee.Algorithms.If(
                ee.Algorithms.IsEqual(cloud_match, None),
                ee.Image.constant(0),
                ee.Image(cloud_match).select("probability"),
            )
        ).rename("cloud_probability")
        return image.addBands(probability_band)

    return ee.ImageCollection(matches).map(_add_probability)


def _mask_sentinel2(image: ee.Image, cloud_prob_max: int) -> ee.Image:
    cloud_probability = image.select("cloud_probability")
    prob_mask = cloud_probability.lte(cloud_prob_max)

    qa60 = image.select("QA60")
    not_cloud = qa60.bitwiseAnd(1 << 10).eq(0)
    not_cirrus = qa60.bitwiseAnd(1 << 11).eq(0)
    qa_mask = not_cloud.And(not_cirrus)

    scl = image.select("SCL")
    shadow_mask = scl.neq(3).And(scl.neq(11))

    combined_mask = prob_mask.And(qa_mask).And(shadow_mask)
    return image.updateMask(combined_mask)


def monthly_sentinel2_collection(
    geometry: ee.Geometry, month: str, cloud_prob_max: int = 40
) -> Tuple[ee.ImageCollection, ee.Image]:
    """Build a masked Sentinel-2 collection and monthly composite for a month string."""
    start_iso, end_iso = month_date_range(month)

    base_collection = (
        ee.ImageCollection(S2_SR_COLLECTION)
        .filterBounds(geometry)
        .filterDate(start_iso, end_iso)
    )

    probability = (
        ee.ImageCollection(S2_CLOUD_PROB_COLLECTION)
        .filterBounds(geometry)
        .filterDate(start_iso, end_iso)
    )

    with_clouds = _attach_cloud_probability(base_collection, probability)
    masked = with_clouds.map(lambda img: _mask_sentinel2(img, cloud_prob_max))

    composite = masked.median().select(S2_BANDS)
    composite = composite.set({"system:time_start": ee.Date(start_iso).millis(), "month": month})
    return masked, composite


def list_collection_images(collection: ee.ImageCollection) -> List[ee.Image]:
    """Return a client-side list of images from ``collection``."""

    try:
        count = int(collection.size().getInfo())
    except Exception:
        return []
    if count <= 0:
        return []

    images: List[ee.Image] = []
    try:
        image_list = collection.toList(count)
    except Exception:
        return []

    for idx in range(count):
        try:
            images.append(ee.Image(image_list.get(idx)))
        except Exception:
            continue
    return images

