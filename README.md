# Vision Backend (Template)

A Cloud Run–ready FastAPI backend template for NDVI/indices via Google Earth Engine.


## NDVI monthly endpoint

The `/api/ndvi/monthly` route computes monthly NDVI statistics for a supplied GeoJSON geometry
and date range. When the `collection` field is omitted it now defaults to the harmonized Sentinel-2
surface reflectance dataset `COPERNICUS/S2_SR_HARMONIZED`, aligning with the rest of the
application.

Example request payload:

```json
{
  "geometry": {"type": "Point", "coordinates": [0, 0]},
  "start": "2023-01-01",
  "end": "2023-12-31"
}
```

This payload succeeds without specifying `collection` or `scale`, relying on the defaults.


## Sentinel-2 index exports

The backend exposes an asynchronous workflow for exporting Sentinel-2 monthly
composites and derived indices as GeoTIFFs. Exports are initiated with a POST
request to `/export/s2/indices` with the following JSON payload:

```json
{
  "aoi_geojson": {"type": "Polygon", "coordinates": [[[149.7, -28.8], [149.8, -28.8], [149.8, -28.7], [149.7, -28.7], [149.7, -28.8]]]},
  "months": ["2024-06", "2024-07"],
  "indices": ["NDVI", "NDRE", "NDMI"],
  "export_target": "zip",
  "aoi_name": "Lot1_RP12345",
  "scale_m": 10,
  "cloud_prob_max": 40
}
```

Supported `export_target` values:

* `zip`: downloads are retrieved from `GET /export/s2/indices/{job_id}/download`
  as a single ZIP archive.
* `gcs`: results are written to the bucket defined by `GEE_GCS_BUCKET`
  (fallback: `GCS_BUCKET`). Each item contains a signed URL when the task
  completes.
* `drive`: assets are exported to Google Drive. The status endpoint returns the
  Drive URLs reported by Earth Engine once the tasks complete.

The export workflow returns a job identifier immediately:

```bash
curl -X POST http://localhost:8000/export/s2/indices \
  -H "Content-Type: application/json" \
  -d @payload.json
```

Responses:

* `POST /export/s2/indices` → `{ "job_id": "...", "state": "running" }`
* `GET  /export/s2/indices/{job_id}/status` → JSON with per-file progress,
  destination URIs and signed URLs when available.
* `GET  /export/s2/indices/{job_id}/download` → ZIP stream when
  `export_target=zip`, or the same JSON status document for Drive/GCS jobs.

### Environment variables

* `GEE_SERVICE_ACCOUNT_JSON` – **required**. Either the raw JSON credentials,
  a base64-encoded JSON string, or a path to a JSON file for the Google Earth
  Engine service account. When unset the application falls back to
  `GOOGLE_APPLICATION_CREDENTIALS`.
* `GEE_GCS_BUCKET` – optional bucket name for Cloud Storage exports. Falls back
  to `GCS_BUCKET` when present.
* `GEE_DRIVE_FOLDER` – optional Google Drive folder name for Drive exports. The
  default folder is `Sentinel2_Indices`.

Earth Engine is initialised on application startup using the credentials from
`GEE_SERVICE_ACCOUNT_JSON`. Missing or invalid credentials will be logged but do
not prevent the API from starting, which is useful for local development.

 
