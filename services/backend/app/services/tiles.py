import json
import os
import pathlib
from typing import Any, Mapping

import ee
from app.services.ee_patches import apply_ee_runtime_patches
from app.services.ee_debug import debug_trace, debug_wrap  # noqa: F401
from ee import ServiceAccountCredentials
from app.services.gcs import download_json, exists
from app.services.image_stats import temporal_stats
from app.services.indices import IndexDefinition, resolve_index

apply_ee_runtime_patches()

# Reuse the same SA method you used for NDVI
SA_EMAIL = os.getenv(
    "EE_SERVICE_ACCOUNT", "ee-agri-worker@baradine-farm.iam.gserviceaccount.com"
)


def _find_or_write_keyfile() -> str:
    p = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if p and os.path.exists(p):
        return p
    for c in [
        "/etc/secrets/ee-key.json",
        "/etc/secrets/google-credentials.json",
        "/opt/render/project/src/ee-key.json",
    ]:
        if os.path.exists(c):
            return c
    key_json = os.getenv("EE_KEY_JSON")
    if key_json:
        tmp = "/tmp/ee-key.json"
        pathlib.Path(tmp).write_text(json.dumps(json.loads(key_json)))
        return tmp
    raise RuntimeError(
        "No EE credentials. Set GOOGLE_APPLICATION_CREDENTIALS (file) or EE_KEY_JSON (env)."
    )


def init_ee():
    creds = ServiceAccountCredentials(SA_EMAIL, _find_or_write_keyfile())
    ee.Initialize(credentials=creds, opt_url="https://earthengine.googleapis.com")


DEFAULT_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
_CLOUD_COVER_THRESHOLD = 60


# ---------- Image builders ----------


def _s2_index_collection(
    geom: ee.Geometry,
    start: str,
    end: str,
    *,
    definition: IndexDefinition,
    parameters: Mapping[str, Any],
    collection: str = DEFAULT_COLLECTION,
) -> ee.ImageCollection:
    """Return a Sentinel-2 image collection with the requested index band mapped."""

    def _map_index_band(img):
        image = ee.Image(img)
        index_band = ee.Image(definition.compute(image, parameters))
        return image.addBands(index_band)

    coll = (
        ee.ImageCollection(collection)
        .filterBounds(geom)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", _CLOUD_COVER_THRESHOLD))
        .map(_map_index_band)
    )
    return coll


def _index_image_for_range(
    geometry_geojson: dict,
    start: str,
    end: str,
    *,
    definition: IndexDefinition,
    parameters: Mapping[str, Any],
    collection: str = DEFAULT_COLLECTION,
) -> ee.Image:
    geom = ee.Geometry(geometry_geojson)

    coll = _s2_index_collection(
        geom,
        start,
        end,
        definition=definition,
        parameters=parameters,
        collection=collection,
    )
    stats = temporal_stats(
        coll.select(definition.band_name),
        band_name=definition.band_name,
        rename_prefix=definition.band_name,
        mean_band_name=definition.band_name,
    )
    mean_img = stats["mean"]
    img = mean_img.clip(geom)
    if definition.valid_range is not None:
        low, high = definition.valid_range
        img = img.clamp(low, high)
    return img


def index_annual_image(
    geometry_geojson: dict,
    year: int,
    *,
    definition: IndexDefinition,
    parameters: Mapping[str, Any],
    collection: str = DEFAULT_COLLECTION,
) -> ee.Image:
    start, end = f"{year}-01-01", f"{year}-12-31"
    return _index_image_for_range(
        geometry_geojson,
        start,
        end,
        definition=definition,
        parameters=parameters,
        collection=collection,
    )


def index_month_image(
    geometry_geojson: dict,
    year: int,
    month: int,
    *,
    definition: IndexDefinition,
    parameters: Mapping[str, Any],
    collection: str = DEFAULT_COLLECTION,
) -> ee.Image:
    start = f"{year}-{month:02d}-01"
    if month == 12:
        end = f"{year + 1}-01-01"
    else:
        end = f"{year}-{month + 1:02d}-01"
    return _index_image_for_range(
        geometry_geojson,
        start,
        end,
        definition=definition,
        parameters=parameters,
        collection=collection,
    )


def ndvi_annual_image(geometry_geojson: dict, year: int) -> ee.Image:
    definition, params = resolve_index("ndvi")
    return index_annual_image(
        geometry_geojson,
        year,
        definition=definition,
        parameters=params,
    )


def ndvi_month_image(geometry_geojson: dict, year: int, month: int) -> ee.Image:
    definition, params = resolve_index("ndvi")
    return index_month_image(
        geometry_geojson,
        year,
        month,
        definition=definition,
        parameters=params,
    )


# ---------- Tile URL factory ----------


def _coerce_palette(value: Any) -> list[str]:
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple)):
        items = [str(part) for part in value if str(part)]
    else:
        raise TypeError("Palette override must be a string or sequence of strings.")

    if not items:
        raise ValueError("Palette override must include at least one colour.")
    return items


def _build_visualization(
    definition: IndexDefinition,
    overrides: Mapping[str, Any] | None = None,
) -> dict:
    vis = definition.default_visualization()
    if overrides:
        vis = dict(vis)  # copy to avoid mutating default
        if "min" in overrides and overrides["min"] is not None:
            vis["min"] = float(overrides["min"])
        if "max" in overrides and overrides["max"] is not None:
            vis["max"] = float(overrides["max"])
        if "palette" in overrides and overrides["palette"] is not None:
            vis["palette"] = _coerce_palette(overrides["palette"])
    return vis


def resolve_clamp_range(
    definition: IndexDefinition,
    overrides: Mapping[str, Any] | None = None,
) -> tuple[float, float] | None:
    base = definition.valid_range
    min_val = base[0] if base is not None else None
    max_val = base[1] if base is not None else None
    if overrides:
        override_min = overrides.get("min")
        override_max = overrides.get("max")
        if override_min is not None:
            min_val = float(override_min)
        if override_max is not None:
            max_val = float(override_max)
    if min_val is None or max_val is None:
        return None
    return float(min_val), float(max_val)


def get_tile_template_for_image(
    img: ee.Image,
    *,
    definition: IndexDefinition,
    vis_overrides: Mapping[str, Any] | None = None,
) -> dict:
    """Return map tile metadata for the provided index image."""

    vis = _build_visualization(definition, vis_overrides)
    mp = img.getMapId(vis)
    tile_url = mp["tile_fetcher"].url_format
    return {
        "mapid": mp["mapid"],
        "token": mp["token"],
        "tile_url": tile_url,
        "vis": vis,
    }


def load_field_geometry(field_id: str) -> dict:
    path = f"fields/{field_id}/field.geojson"
    if not exists(path):
        raise FileNotFoundError("Field geometry not found in GCS.")
    return download_json(path)
