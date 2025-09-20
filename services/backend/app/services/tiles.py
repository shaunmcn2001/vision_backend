import ee, os, json, pathlib
from ee import ServiceAccountCredentials
from app.services.gcs import download_json, exists

# Reuse the same SA method you used for NDVI
SA_EMAIL = os.getenv("EE_SERVICE_ACCOUNT", "ee-agri-worker@baradine-farm.iam.gserviceaccount.com")

def _find_or_write_keyfile() -> str:
    p = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if p and os.path.exists(p):
        return p
    for c in ["/etc/secrets/ee-key.json", "/etc/secrets/google-credentials.json", "/opt/render/project/src/ee-key.json"]:
        if os.path.exists(c):
            return c
    key_json = os.getenv("EE_KEY_JSON")
    if key_json:
        tmp = "/tmp/ee-key.json"
        pathlib.Path(tmp).write_text(json.dumps(json.loads(key_json)))
        return tmp
    raise RuntimeError("No EE credentials. Set GOOGLE_APPLICATION_CREDENTIALS (file) or EE_KEY_JSON (env).")

def init_ee():
    creds = ServiceAccountCredentials(SA_EMAIL, _find_or_write_keyfile())
    ee.Initialize(credentials=creds, opt_url="https://earthengine.googleapis.com")

# ---------- Image builders ----------

def _s2_ndvi_collection(geom: ee.Geometry, start: str, end: str, collection: str = "COPERNICUS/S2_SR_HARMONIZED"):
    coll = (ee.ImageCollection(collection)
            .filterBounds(geom)
            .filterDate(start, end)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))
            .map(lambda img: img.addBands(img.normalizedDifference(["B8", "B4"]).rename("NDVI"))))
    return coll

def ndvi_annual_image(geometry_geojson: dict, year: int) -> ee.Image:
    geom = ee.Geometry(geometry_geojson)
    start, end = f"{year}-01-01", f"{year}-12-31"
    coll = _s2_ndvi_collection(geom, start, end)
    ndvi_band = (
        coll.select("NDVI")
        .mean()
        .resample("bilinear")
        .reproject("EPSG:4326", None, 10)
    )
    img = ndvi_band.clip(geom)
    # Clamp NDVI values to the theoretical range so negatives are preserved
    return img.clamp(-1, 1)

def ndvi_month_image(geometry_geojson: dict, year: int, month: int) -> ee.Image:
    geom = ee.Geometry(geometry_geojson)
    start = f"{year}-{month:02d}-01"
    # safe end-of-month: add 32 days and set to day 1 previous; simpler: next month start
    if month == 12:
        end = f"{year+1}-01-01"
    else:
        end = f"{year}-{month+1:02d}-01"
    coll = _s2_ndvi_collection(geom, start, end)
    ndvi_band = (
        coll.select("NDVI")
        .mean()
        .resample("bilinear")
        .reproject("EPSG:4326", None, 10)
    )
    img = ndvi_band.clip(geom)
    return img.clamp(-1, 1)

# ---------- Tile URL factory ----------

DEFAULT_PALETTE = [
    "440154","482173","433E85","38598C","2D708E","25858E","1E9B8A","2BB07F","51C56A","85D54A","C2DF23","FDE725"
]  # viridis-ish hex (no '#')

def get_tile_template_for_image(img: ee.Image, vis: dict | None = None) -> dict:
    """
    Returns a dict containing {mapid, token, tile_url, vis} compatible with XYZ tile layers.
    """
    if vis is None:
        vis = {"bands": ["NDVI"], "min": -1.0, "max": 1.0, "palette": DEFAULT_PALETTE}
    mp = img.getMapId(vis)  # legacy-style, still supported by earthengine-api
    # mp contains: {'mapid': ..., 'token': ..., 'tile_fetcher': ...}
    tile_url = mp["tile_fetcher"].url_format  # e.g. https://earthengine.googleapis.com/map/{mapid}/{z}/{x}/{y}?token={token}
    return {"mapid": mp["mapid"], "token": mp["token"], "tile_url": tile_url, "vis": vis}

def load_field_geometry(field_id: str) -> dict:
    path = f"fields/{field_id}/field.geojson"
    if not exists(path):
        raise FileNotFoundError("Field geometry not found in GCS.")
    return download_json(path)
