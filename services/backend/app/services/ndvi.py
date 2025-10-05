from __future__ import annotations

import csv
import io
import json
import os
import pathlib
from datetime import date
from typing import Any, Dict, Mapping

import ee
from ee import ServiceAccountCredentials

from app.services.gcs import (
    _bucket,
    download_json,
    exists,
    list_prefix,
    upload_json,
)
from app.services.image_stats import temporal_stats
from app.services.indices import normalize_index_code, resolve_index


DEFAULT_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
DEFAULT_SCALE = 10
SA_EMAIL = os.getenv(
    "EE_SERVICE_ACCOUNT", "ee-agri-worker@baradine-farm.iam.gserviceaccount.com"
)


def _find_or_write_keyfile() -> str:
    p = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if p and os.path.exists(p):
        return p
    for candidate in (
        "/etc/secrets/ee-key.json",
        "/etc/secrets/google-credentials.json",
        "/opt/render/project/src/ee-key.json",
    ):
        if os.path.exists(candidate):
            return candidate
    key_json = os.getenv("EE_KEY_JSON")
    if key_json:
        tmp = "/tmp/ee-key.json"
        pathlib.Path(tmp).write_text(json.dumps(json.loads(key_json)))
        return tmp
    raise RuntimeError(
        "No EE credentials. Set GOOGLE_APPLICATION_CREDENTIALS (file) or EE_KEY_JSON (env)."
    )


def init_ee():
    keyfile = _find_or_write_keyfile()
    creds = ServiceAccountCredentials(SA_EMAIL, keyfile)
    ee.Initialize(credentials=creds, opt_url="https://earthengine.googleapis.com")


def compute_monthly_index(
    geometry: dict,
    *,
    start: date,
    end: date,
    index_code: str,
    collection: str = DEFAULT_COLLECTION,
    scale: int = DEFAULT_SCALE,
    parameters: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    if end < start:
        raise ValueError("End date must be on or after start date.")

    init_ee()

    geom = ee.Geometry(geometry)
    definition, resolved_params = resolve_index(index_code, parameters)

    mapped_collection = (
        ee.ImageCollection(collection)
        .filterBounds(geom)
        .filterDate(start.isoformat(), end.isoformat())
        .map(lambda img: img.addBands(definition.compute(img, resolved_params)))
    )

    values = []
    for month in _iterate_months(start, end):
        monthly_collection = mapped_collection.filter(
            ee.Filter.calendarRange(month, month, "month")
        )
        stats = temporal_stats(
            monthly_collection,
            band_name=definition.band_name,
            rename_prefix=definition.band_name,
            mean_band_name=definition.band_name,
        )
        monthly_mean = stats["mean"].select(definition.band_name)
        result = monthly_mean.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=scale,
            bestEffort=True,
        ).get(definition.band_name)
        value = result.getInfo() if result is not None else None
        if value is None:
            continue
        values.append({"month": int(month), definition.code: value})

    return {
        "index": {
            "code": definition.code,
            "band": definition.band_name,
            "valid_range": list(definition.valid_range)
            if definition.valid_range is not None
            else None,
            "parameters": dict(resolved_params),
        },
        "data": values,
    }


def compute_monthly_index_for_year(
    geometry: dict,
    year: int,
    *,
    index_code: str,
    collection: str = DEFAULT_COLLECTION,
    scale: int = DEFAULT_SCALE,
    parameters: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    start = date(year, 1, 1)
    end = date(year, 12, 31)
    return compute_monthly_index(
        geometry,
        start=start,
        end=end,
        index_code=index_code,
        collection=collection,
        scale=scale,
        parameters=parameters,
    )


def gcs_index_prefix(index_code: str, field_id: str) -> str:
    normalized = normalize_index_code(index_code)
    return f"index-results/{normalized}/{field_id}"


def gcs_index_path(index_code: str, field_id: str, year: int) -> str:
    return f"{gcs_index_prefix(index_code, field_id)}/{year}.json"


def gcs_index_csv_path(index_code: str, field_id: str, year: int) -> str:
    return f"{gcs_index_prefix(index_code, field_id)}/{year}.csv"


def get_or_compute_and_cache_index(
    field_id: str,
    geometry: dict,
    year: int,
    *,
    index_code: str,
    collection: str = DEFAULT_COLLECTION,
    scale: int = DEFAULT_SCALE,
    parameters: Mapping[str, Any] | None = None,
    force: bool = False,
) -> Dict[str, Any]:
    path = gcs_index_path(index_code, field_id, year)
    if not force and exists(path):
        return download_json(path)

    result = compute_monthly_index_for_year(
        geometry,
        year,
        index_code=index_code,
        collection=collection,
        scale=scale,
        parameters=parameters,
    )

    payload = {
        "field_id": field_id,
        "year": year,
        "index": result["index"],
        "data": result["data"],
    }
    upload_json(payload, path)

    csv_path = gcs_index_csv_path(index_code, field_id, year)
    upload_index_csv(result["data"], csv_path, result["index"]["code"])

    return payload


def upload_index_csv(rows: list[dict], path: str, index_key: str):
    bucket = _bucket()
    blob = bucket.blob(path)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["month", index_key])
    writer.writeheader()
    writer.writerows(rows)
    blob.cache_control = "no-cache"
    blob.upload_from_string(buffer.getvalue(), content_type="text/csv")


def list_cached_years(field_id: str, index_code: str) -> list[int]:
    prefix = gcs_index_prefix(index_code, field_id)
    names = list_prefix(f"{prefix}/")
    years: set[int] = set()
    for name in names:
        fname = name.split("/")[-1]
        if fname.endswith(".json"):
            try:
                years.add(int(fname[:-5]))
            except ValueError:
                continue
    return sorted(years)


def _iterate_months(start: date, end: date) -> list[int]:
    months: list[int] = []
    year = start.year
    month = start.month
    while (year, month) <= (end.year, end.month):
        months.append(month)
        month += 1
        if month > 12:
            month = 1
            year += 1
    return months


__all__ = [
    "DEFAULT_COLLECTION",
    "DEFAULT_SCALE",
    "compute_monthly_index",
    "compute_monthly_index_for_year",
    "get_or_compute_and_cache_index",
    "gcs_index_csv_path",
    "gcs_index_path",
    "gcs_index_prefix",
    "init_ee",
    "list_cached_years",
    "upload_index_csv",
]
