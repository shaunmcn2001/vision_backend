# Vision

Vision delivers Earth Engine–powered NDVI products through a FastAPI backend and a modern Vite + React dashboard. The service focuses on browser-first delivery: every export is exposed via direct Earth Engine download URLs (GeoTIFF, SHP, CSV) with no Google Drive or Cloud Storage staging.

## Project layout

```
services/backend/    FastAPI application (Python 3.11)
frontend/            Vite + React + Tailwind + shadcn/ui interface
```

## Environment & configuration

Set the following environment variables before running the backend:

```
GCP_PROJECT=baradine-farm
GCP_REGION=australia-southeast1
GOOGLE_APPLICATION_CREDENTIALS=/path/to/ee-service-account.json
CORS_ORIGINS=https://your-frontend.example.com
```

The service account must be enabled for Google Earth Engine. All exports are streamed via Earth Engine `getDownloadURL` calls and are capped by `MAX_PIXELS_FOR_DIRECT` (default `3e7`).

## Backend setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r services/backend/requirements.txt
cd services/backend
uvicorn app.main:app --reload
```

### Key endpoints

- `POST /api/ndvi-month` – monthly NDVI composites and a mean preview tile.
- `POST /api/imagery/daily` – daily Sentinel-2 RGB mosaics with `cloudPct` analytics.
- `POST /api/zones/basic` – five-class NDVI quantile zones with GeoTIFF/SHP/CSV downloads.
- `POST /api/zones/advanced` – stage-aware long-term zones with raster/vector/stat CSVs.
- `POST /api/tiles/session` + `GET /api/tiles/{token}/{z}/{x}/{y}` – tile session utilities used by the UI.

Example NDVI Month request:

```bash
curl -X POST http://localhost:8000/api/ndvi-month \
  -H "Content-Type: application/json" \
  -d '{
        "aoi": {"type": "Polygon", "coordinates": [[[149.5,-31.1],[149.6,-31.1],[149.6,-31.0],[149.5,-31.0],[149.5,-31.1]]]},
        "start": "2024-05-01",
        "end": "2024-08-31",
        "clamp": [0,1]
      }'
```

Each product response returns tile tokens for map previews and direct download URLs for rasters, vectors, and CSV summaries when applicable. Oversized AOIs trigger HTTP 400 with guidance to reduce the extent/date range.

## Frontend setup

```bash
cd frontend
npm install
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

Set `VITE_API_BASE_URL` to the backend origin for local development. The UI provides:

- Product cards for NDVI Month, Daily Imagery, Basic NDVI Zones, and Advanced Zones.
- AOI input via shapefile ZIP upload or GeoJSON paste.
- Date range selection, clamp controls, bands selection, and season CSV ingestion.
- MapLibre map with AOI outlines and tile overlays.
- Results panel with per-day imagery summaries and download link listings.

## Testing

Backend tests:

```bash
cd services/backend
pytest
```

Frontend tests:

```bash
cd frontend
npm test
```

## Deployment notes

- **Backend (Render/Cloud Run):** Supply the Earth Engine service account JSON via secret volume or environment file and expose HTTP on port `8000`.
- **Frontend (Render static site / Netlify / Vercel):** Build with `npm run build` and serve `/dist`. Configure `VITE_API_BASE` or rely on same origin with a reverse proxy.
- Ensure public endpoints are protected (API keys, gateway) before exposing beyond trusted clients.

## Direct download behaviour

All exports rely on Earth Engine direct download URLs (`Image.getDownloadURL`, `FeatureCollection.getDownloadURL`). Downloads are initiated in the browser and never staged in Google Drive or GCS. Large AOIs beyond the configured pixel guard are rejected early with an actionable error.
