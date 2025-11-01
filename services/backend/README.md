# Vision Backend

FastAPI service that wraps Google Earth Engine (GEE) workflows for Sentinel-2 export jobs, NDVI analytics, and field boundary storage. The service is ready for Cloud Run deployment but can be run locally with a service account.

## Features
- **Sentinel-2 index exports** – queue monthly composites, download ZIPs, or launch Drive/GCS tasks.
- **Field ingestion & storage** – validate shapefile/KML uploads and persist metadata to GCS.
- **NDVI analytics** – compute and cache statistics with CSV downloads.
- **Tile services** – deliver map tiles for composites and zone previews.

## Getting Started

### Prerequisites
- Python 3.10+
- Google Earth Engine service account credentials (JSON file, raw JSON, or base64 string)
- Optional: Google Cloud Storage bucket and Drive folder for exports

### Installation
```bash
cd services/backend
python -m venv .venv
source .venv/bin/activate           # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Environment
Copy `.env.example` to `.env` (or inject variables via your deployment platform) before running the API:

```
cp services/backend/.env.example services/backend/.env
```

Set or override the following variables as needed:

| Variable | Description |
|----------|-------------|
| `API_KEY` | Required header value for protected routes. |
| `BACKFILL_START` | Earliest date for NDVI backfill jobs. |
| `CLOUD_SQL_INSTANCE` | Cloud SQL instance connection string. |
| `CLOUD_TASKS_QUEUE` | Cloud Tasks queue name for background jobs. |
| `CORS_ORIGINS` | Allowed origins for CORS (comma separated or `*`). |
| `DB_NAME` / `DB_USER` / `DB_PASS` | Database credentials for the Cloud SQL instance. |
| `DEBUG` | Enable verbose logging when `true`. |
| `GCP_PROJECT` / `GCP_REGION` | GCP project metadata used for jobs and storage. |
| `GCS_BUCKET` | Bucket for exports and NDVI cache. |
| `GEE_SERVICE_ACCOUNT_JSON` | Raw JSON, base64 string, or path to the Earth Engine key. |
| `GEE_DRIVE_FOLDER` | Drive folder name for export tasks (optional). |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to the credential file when mounted on disk. |
| `MIN_FIELD_HA` | Minimum field area in hectares (default from config). |

### Run Locally
```bash
cd services/backend
uvicorn app.main:app --reload
```
The API and `/ui` console will be available at `http://localhost:8000`.

### Tests
```bash
cd services/backend
pytest
```
Tests rely on bundled fakes and do not hit live Earth Engine.

## Key Directories
- `app/api/` – FastAPI routers for products, tiles, and uploads.
- `app/services/` – Earth Engine helpers, downloads, and NDVI logic.
- `app/utils/` – Shapefile/KML parsing plus geometry utilities.
- `app/exports.py` – Background job orchestration and download staging.
- `tests/` – Pytest suite with `fake_ee.py` for deterministic runs.

## Deployment Notes
- The service is container-friendly; see `Docker/` for Cloud Run images.
- The top-level `Procfile` mirrors the production Uvicorn command.
- Ensure credentials are supplied via environment variables or mounted volumes when deploying.
