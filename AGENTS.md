# Vision Backend contributor guide

## Repository layout
- `services/backend/app/` contains the FastAPI application package.
  - `main.py` configures the ASGI app, middleware, and startup hooks.
  - `api/` exposes REST endpoints for Sentinel-2 exports, NDVI stats, tiles, and field uploads.
  - `exports.py` orchestrates Sentinel-2 export jobs, background polling, and cleanup across GCS/Drive/ZIP targets.
  - `gee.py` wraps Google Earth Engine initialisation, geometry parsing, and Sentinel-2 collection helpers.
  - `indices.py`, `services/`, and `utils/` provide reusable vegetation index math, service layer helpers, and shapefile/GeoJSON utilities used by the API.
- `services/backend/tests/` hosts the pytest suite. It supplies fake Earth Engine primitives (`fake_ee.py`) and integration-style API tests that simulate export flows and NDVI requests.
- `services/backend/Docker/` holds container entrypoints and scripts for deployment; `Procfile` mirrors the production Uvicorn command.

## Coding conventions
- Target Python 3.10+ with type hints; many modules use `from __future__ import annotations` and dataclasses.
- Keep business logic in `app/services`/`app/utils` and reserve FastAPI routers for request/response handling.
- Prefer small, pure functions that are easy to exercise via the existing unit tests and fake Earth Engine doubles.
- When touching export orchestration, preserve the job-registry locking discipline in `app/exports.py`.

## Environment configuration
- Runtime and tests rely on Google Earth Engine credentials via `GEE_SERVICE_ACCOUNT_JSON` (raw JSON string or a path). Optional targets: `GEE_GCS_BUCKET` for Cloud Storage exports and `GEE_DRIVE_FOLDER` for Drive exports.
- Local development typically installs dependencies with `pip install -r services/backend/requirements.txt` plus `pip install pytest` for the test harness.

## Tests
- Run the Python suite from the backend service root so the `app` package resolves correctly:
  ```bash
  cd services/backend
  pytest
  ```
  The tests monkeypatch Earth Engine calls and network requests, so no live GEE access is required.
- CI expectations mirror the above `pytest` run; add new tests under `services/backend/tests/` alongside the existing fakes when expanding functionality.
