import ee
from ee import ServiceAccountCredentials
import os, json, pathlib
from app.services.gcs import upload_json, download_json, exists

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
    keyfile = _find_or_write_keyfile()
    creds = ServiceAccountCredentials(SA_EMAIL, keyfile)
    ee.Initialize(credentials=creds, opt_url="https://earthengine.googleapis.com")

def compute_monthly_ndvi(geometry: dict, year: int, collection: str = "COPERNICUS/S2_SR_HARMONIZED", scale: int = 10):
    init_ee()
    geom = ee.Geometry(geometry)
    coll = (ee.ImageCollection(collection)
            .filterBounds(geom)
            .filterDate(f"{year}-01-01", f"{year}-12-31")
            .map(lambda img: img.addBands(img.normalizedDifference(["B8","B4"]).rename("NDVI"))))
    months = ee.List.sequence(1, 12)
    out = []
    for m in months.getInfo():
        mean = coll.filter(ee.Filter.calendarRange(m, m, "month")).mean().select("NDVI")
        val = mean.reduceRegion(reducer=ee.Reducer.mean(), geometry=geom, scale=scale, bestEffort=True).get("NDVI").getInfo()
        out.append({"month": int(m), "ndvi": val})
    return out

def gcs_ndvi_path(field_id: str, year: int) -> str:
    return f"ndvi-results/{field_id}/{year}.json"

def get_or_compute_and_cache_ndvi(field_id: str, geometry: dict, year: int, force: bool = False) -> dict:
    path = gcs_ndvi_path(field_id, year)
    if not force and exists(path):
        return download_json(path)
    data = compute_monthly_ndvi(geometry, year)
    payload = {"field_id": field_id, "year": year, "data": data}
    upload_json(payload, path)
    return payload
