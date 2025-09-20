import ee
from ee import ServiceAccountCredentials
import os, json, pathlib
import csv, io
from app.services.gcs import upload_json, download_json, exists, _bucket, list_prefix


SA_EMAIL = os.getenv("EE_SERVICE_ACCOUNT", "ee-agri-worker@baradine-farm.iam.gserviceaccount.com")

DEFAULT_REDUCE_REGION_SCALE = 10
DEFAULT_REDUCE_REGION_CRS = "EPSG:3857"

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

def reduce_region_sampling(
    scale: float = DEFAULT_REDUCE_REGION_SCALE,
    crs: str | None = DEFAULT_REDUCE_REGION_CRS,
    *,
    best_effort: bool = True,
) -> dict:
    """Build consistent ``reduceRegion`` sampling kwargs for 10 m statistics."""
    kwargs: dict = {"scale": scale}
    if crs:
        kwargs["crs"] = crs
    if best_effort is not None:
        kwargs["bestEffort"] = best_effort
    return kwargs

def compute_monthly_ndvi(
    geometry: dict,
    year: int,
    collection: str = "COPERNICUS/S2_SR_HARMONIZED",
    scale: float = DEFAULT_REDUCE_REGION_SCALE,
    crs: str | None = DEFAULT_REDUCE_REGION_CRS,
):
    init_ee()
    geom = ee.Geometry(geometry)
    coll = (ee.ImageCollection(collection)
            .filterBounds(geom)
            .filterDate(f"{year}-01-01", f"{year}-12-31")
            .map(lambda img: img.addBands(img.normalizedDifference(["B8","B4"]).rename("NDVI"))))
    months = ee.List.sequence(1, 12)
    out = []
    sampling_kwargs = reduce_region_sampling(scale=scale, crs=crs)
    for m in months.getInfo():
        mean = coll.filter(ee.Filter.calendarRange(m, m, "month")).mean().select("NDVI")
        val = mean.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            **sampling_kwargs,
        ).get("NDVI").getInfo()
        out.append({"month": int(m), "ndvi": val})
    return out

def gcs_ndvi_path(field_id: str, year: int) -> str:
    return f"ndvi-results/{field_id}/{year}.json"

def get_or_compute_and_cache_ndvi(
    field_id: str,
    geometry: dict,
    year: int,
    force: bool = False,
    *,
    scale: float = DEFAULT_REDUCE_REGION_SCALE,
    crs: str | None = DEFAULT_REDUCE_REGION_CRS,
) -> dict:
    path = gcs_ndvi_path(field_id, year)
    if not force and exists(path):
        return download_json(path)
    data = compute_monthly_ndvi(geometry, year, scale=scale, crs=crs)
    payload = {"field_id": field_id, "year": year, "data": data}
    upload_json(payload, path)
    csv_path = f"ndvi-results/{field_id}/{year}.csv"
    upload_csv(data, csv_path)
    return payload

def upload_csv(rows: list[dict], path: str):
    bucket = _bucket()
    blob = bucket.blob(path)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["month","ndvi"])
    w.writeheader()
    w.writerows(rows)
    blob.cache_control = "no-cache"
    blob.upload_from_string(buf.getvalue(), content_type="text/csv")

def list_cached_years(field_id: str) -> list[int]:
    """
    Inspect GCS under ndvi-results/{field_id}/ and return the years that have a *.json cache.
    """
    prefix = f"ndvi-results/{field_id}/"
    names = list_prefix(prefix)  # e.g. ['ndvi-results/<id>/2019.json', '.../2020.json']
    years: set[int] = set()
    for n in names:
        fname = n.split("/")[-1]
        if fname.endswith(".json"):
            try:
                years.add(int(fname[:-5]))  # strip '.json'
            except ValueError:
                pass
    return sorted(years)
