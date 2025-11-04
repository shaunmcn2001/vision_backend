# Vision Backend contributor guide

> **Agent protocol:** Every agent who modifies the repository **must** append an entry to the “Agent activity log” at the bottom of this file before finishing their work. Summaries should include the date, author, and a short description of the changes that landed. Do not leave this file out of sync with your contribution.

## Overview
This repository hosts a Cloud Run–ready FastAPI service that wraps a suite of
Google Earth Engine (GEE) workflows for agriculture analytics. The backend now
covers three pillars:

1. **Sentinel-2 monthly export pipeline** – accepts zipped shapefiles, queues
   per-month composites for a list of vegetation indices, and delivers the
   GeoTIFFs as a browser download, Cloud Storage objects, or Google Drive tasks.
2. **Field management** – persists uploaded field boundaries (ZIP/KML/KMZ) and
   exposes CRUD-style endpoints backed by GCS.
3. **Index analytics & tiling** – computes monthly NDVI/indices statistics,
   caches results to GCS/CSV, and generates tile URLs for mapping clients.

The current codebase stitches together multiple updates that were required to
arrive at this working app:

- Hardened Earth Engine initialisation that accepts raw JSON, base64 strings, or
  file paths (`app/gee.py`). The FastAPI startup hook logs failures but keeps the
  API responsive for local work.
- A dedicated `app/exports.py` job runner with per-job locking, TTL-based
  eviction, sanitised AOI names, on-disk staging for ZIP downloads, and optional
  Cloud Storage / Drive task creation. A lightweight thread pool processes jobs
  in the background.
- REST endpoints under `app/api/s2_indices.py` that orchestrate the job lifecycle
  (upload AOI, queue export, poll status, download results) and enforce area and
  input validation. `/ui` serves a self-contained HTML+JS client that wires these
  endpoints together and reports progress via polling.
- Geometry ingestion hardened via `app/utils/shapefile.py`, which safely extracts
  shapefile members, merges polygons, and rejects archives with path traversal or
  missing `.shp` members. KML/KMZ parsing support lives in
  `app/api/fields_upload.py` alongside area checks powered by
  `app/utils/geometry.py` (equal-area projection for hectares).
- Field metadata plus AOI geometries are stored in Google Cloud Storage using the
  helpers inside `app/services/gcs.py`. Every field upload updates a lightweight
  `fields/index.json` catalogue for quick listing.
- Monthly NDVI and companion indices share parameterised definitions in
  `app/services/indices.py`. `app/services/ndvi.py` (and the `/api/ndvi/*`
  endpoints) either stream fresh computations from GEE or reuse cached JSON/CSV
  artefacts in GCS. The default dataset is now the harmonised
  `COPERNICUS/S2_SR_HARMONIZED` collection, matching the export workflow.
- `app/services/tiles.py` plus `app/api/tiles.py` provide map tile templates for
  annual or monthly composites with optional visualisation overrides and clamping
  to valid index ranges.
- A minimal API key gate lives in `app/main.py`: everything other than the root
  page, `/ui`, health checks, and docs requires an `x-api-key` header that matches
  `API_KEY`.

Together these pieces replaced earlier stubs with a production-like flow that can
be exercised end-to-end via the bundled UI or HTTP clients.

## Repository layout
- `services/backend/app/` contains the FastAPI application package.
  - `main.py` wires middleware, startup hooks, routers, the `/ui` export console,
    and API-key enforcement.
  - `api/` hosts routers for Sentinel-2 exports, NDVI stats, tile metadata, and
    field ingestion.
  - `exports.py` orchestrates Sentinel-2 jobs, handles download staging, Cloud
    Storage/Drive export tasks, job cleanup, and signed URL generation.
  - `gee.py` normalises credential handling, exposes helpers for month ranges,
    and builds masked Sentinel-2 composites.
  - `indices.py`, `services/`, and `utils/` provide index math, service-layer
    wrappers for GCS/Earth Engine interactions, shapefile/KML ingestion, and
    geometry utilities.
- `services/backend/tests/` supplies the pytest suite with a `fake_ee.py` helper
  that simulates Earth Engine collections/images. Tests cover shapefile parsing,
  export registry behaviour, NDVI calculations, tile endpoints, and upload flows.
- `services/backend/Docker/` holds deployment scripts and entrypoints; the
  top-level `Procfile` mirrors the production Uvicorn command.

## Key application flows

### Sentinel-2 index exports
1. **AOI preparation** – `POST /export/s2/indices/aoi` accepts a shapefile ZIP,
   validates the geometry and area threshold, and returns GeoJSON plus a
   sanitised AOI name. The same sanitisation is used later for filenames.
2. **Job creation** – `POST /export/s2/indices` validates month strings,
   supported index names (currently the 20+ entries defined in `app/indices.py`),
   and destination (`zip`, `gcs`, or `drive`). `exports.create_job` builds an
   `ExportJob` with per-month/per-index `ExportItem`s, then queues `_run_job` on
   a thread pool.
3. **Background processing** – `_run_job` initialises GEE, builds monthly
   Sentinel-2 composites via `gee.monthly_sentinel2_collection`, computes each
   requested index (`app/indices.compute_index`), and hands off to
   `_process_zip_exports` or `_process_cloud_exports`. Cloud exports respect
   `MAX_CONCURRENT_EXPORTS` and poll Earth Engine tasks until completion/failure,
   attaching signed URLs when possible.
4. **Status & retention** – `GET /export/s2/indices/{job_id}/status` surfaces
   job progress. Completed jobs remain in memory for 24h, after which they are
   evicted (with early removal when files are cleaned). Evicted IDs are tracked
   to return HTTP 410 responses instead of 404 when clients poll too late.
5. **Download** – `GET /export/s2/indices/{job_id}/download` streams a ZIP for
   `zip` jobs (and triggers temp file cleanup via `BackgroundTask`). For Drive
   or GCS destinations the same endpoint returns the status payload with
   destination URIs and signed URLs.
6. **UI client** – `/ui` embeds HTML/JS that manages uploads, month derivation,
   job polling, ZIP downloads, and logs Drive/GCS destinations to the browser
   console. The frontend mirrors API validation messages for a friendlier UX.

### Field ingestion & storage
- `POST /api/fields` creates new field records from raw GeoJSON, enforcing a
  configurable minimum area (`MIN_FIELD_HA`). Metadata and geometry are stored in
  `fields/{id}/meta.json` and `fields/{id}/field.geojson` inside the configured
  bucket; the index listing is maintained best-effort.
- `POST /api/fields/upload` accepts shapefile ZIP, KML, or KMZ uploads, reusing
  the shapefile/KML parsers to generate GeoJSON before delegating to the same
  GCS persistence logic. Returned payloads include the generated ID, name, area,
  and timestamp.
- `GET /api/fields` and `/api/fields/{id}` read the index and per-field files to
  provide summaries or full metadata + geometry.

### NDVI and index analytics
- `/api/ndvi/monthly` ingests arbitrary GeoJSON plus a date range, normalises
  index selection (defaults to NDVI), and leverages
  `app/services/ndvi.compute_monthly_index` to map each month via Earth Engine.
  The helper enforces chronological start/end, injects the selected index band,
  and reduces the composite to a mean value.
- `/api/ndvi/monthly/by-field/{field_id}` loads stored field geometries, reuses
  cached JSON if present, and otherwise triggers a computation + upload via
  `get_or_compute_and_cache_index`. CSV artefacts are written alongside the JSON
  using `upload_index_csv` so downstream analysts can fetch tabular data.
- `/api/ndvi/cache` and `/api/ndvi/links` surface cached artefacts (including
  signed URLs generated by `app/services/gcs.sign_url`).

### Map tiles
- `/api/tiles/ndvi/annual` and `/api/tiles/ndvi/month` accept optional
  visualisation overrides (`min`, `max`, `palette`) and assemble tile metadata by
  requesting `getMapId` on the computed image. `resolve_clamp_range` ensures
  colour ramps clamp to valid index ranges when provided.

## Environment configuration
- Earth Engine credentials must be supplied through `GEE_SERVICE_ACCOUNT_JSON`
  (raw JSON string, base64-encoded JSON, or a path). The fallback is
  `GOOGLE_APPLICATION_CREDENTIALS`.
- Optional exports configuration:
  - `GEE_GCS_BUCKET` (fallback `GCS_BUCKET`) for Cloud Storage destinations and
    NDVI caching.
  - `GEE_DRIVE_FOLDER` for Drive exports (defaults to `Sentinel2_Indices`).
  - `MIN_FIELD_HA` to enforce a minimum area threshold when ingesting fields/AOIs.
  - `API_KEY` to enable the middleware check in `app/main.py`.
- The FastAPI app can be launched with `uvicorn app.main:app --reload` from
  `services/backend`. Startup initialises GEE and registers routers.

## Coding conventions
- Target Python 3.10+ with type hints; many modules use `from __future__ import
  annotations` and dataclasses.
- Keep business logic in `app/services`/`app/utils` and reserve FastAPI routers
  for request/response handling.
- Prefer small, pure functions that are easy to exercise via the existing unit
  tests and fake Earth Engine doubles.
- When touching export orchestration, preserve the job-registry locking
  discipline in `app/exports.py`.

## Tests
- Run the Python suite from the backend service root so the `app` package
  resolves correctly:
  ```bash
  cd services/backend
  pytest
  ```
  The tests monkeypatch Earth Engine calls and network requests, so no live GEE
  access is required. Integration-style tests cover the export queues, status
  lifecycle, shapefile parsing, and NDVI calculations.
- CI expectations mirror the above `pytest` run; add new tests under
  `services/backend/tests/` alongside the existing fakes when expanding
  functionality.

## Agent activity log

| Date (UTC) | Agent | Summary |
|------------|-------|---------|
| 2025-11-01 | Codex Agent | Tidied repository, added frontend/backend READMEs, root gitignore, and documented the agent update protocol. |
| 2025-11-01 | Codex Agent | Added backend .env template, documented configuration variables, and recorded required GCP/SQL settings. |
| 2025-11-02 | Codex Agent | Updated backend Docker image to include Google Cloud CLI and noted the change in deployment docs. |
| 2025-11-02 | Codex Agent | Expanded the root README with backend/frontend run instructions and Docker usage. |
| 2025-11-03 | Codex Agent | Loaded the .env file during FastAPI startup to expose credential environment variables. |
| 2025-11-03 | Codex Agent | Updated GCP project settings to use videre-477011 across configuration examples. |
| 2025-11-03 | Codex Agent | Normalised AOI geometry parsing to accept FeatureCollection inputs for Earth Engine requests. |
| 2025-11-03 | Codex Agent | Forced tile proxy requests to use certifi CA bundle to avoid SSL errors. |
| 2025-11-03 | Codex Agent | Expanded tile session zoom range to 0–22 and improved tile proxy TLS handling/logging. |
| 2025-11-03 | Codex Agent | Stored Earth Engine tile fetchers in sessions so tile proxy requests stop failing with 500 errors. |
| 2025-11-03 | Codex Agent | Updated default NDVI colour ramp to match the classic Earth Engine greens/yellows and widened the value range. |
| 2025-11-03 | Codex Agent | Added raw and coloured NDVI GeoTIFF exports with UI download links and regression tests. |
| 2025-11-03 | Codex Agent | Reworked workflow sidebar to show per-product forms (NDVI, imagery, basic & advanced zones) with integrated inputs and shared layout. |
| 2025-11-03 | Codex Agent | Guarded zones percentile lookup to fall back to band-prefixed keys and return 400 when Earth Engine masks out data. |
| 2025-11-03 | Codex Agent | Updated shapefile/CSV download helpers to use Earth Engine's newer getDownloadURL signature (with TypeError fallback). |
| 2025-11-03 | Codex Agent | Normalised zone statistics to recognise new percentile key names so table exports stop failing. |
| 2025-11-03 | Codex Agent | Added smoothed zone vectors, configurable class counts, synced colour palettes, and map legends with vector overlays. |
