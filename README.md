# Vision Backend (Template)

A Cloud Run–ready FastAPI backend template for NDVI/indices via Google Earth Engine.

## Dependencies

Install the backend requirements to pull in third-party services, including the real ``rasterio`` distribution used by
the zone-classification pipeline::

    pip install -r services/backend/requirements.txt

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

Supplying a `production_zones` object enables the optional zone export pipeline.
When any zone options are provided the feature is automatically activated, so the
following payload will compute five production classes using the supplied
minimum mapping unit:

```json
{
  "production_zones": {
    "method": "ndvi_kmeans",
    "n_classes": 5,
    "mmu_ha": 3
  }
}
```

Explicitly set `"enabled": false` within the object to opt out while still
including option overrides.

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

### `/zones/production` – production zone exports

Use `POST /zones/production` to build NDVI-based production zones for a period
of months. The endpoint accepts either an explicit `months[]` array or a
`start_month`/`end_month` range which is expanded server-side:

*Explicit month selection*

```bash
curl -X POST http://localhost:8000/api/zones/production \
  -H "Content-Type: application/json" \
  -d '{
        "aoi_geojson": {"type": "Polygon", "coordinates": [[[149.7, -28.8], [149.8, -28.8], [149.8, -28.7], [149.7, -28.7], [149.7, -28.8]]]},
        "aoi_name": "Lot1_RP12345",
        "months": ["2024-03", "2024-04", "2024-05"],
        "n_classes": 5,
        "cv_mask_threshold": 0.25,
        "mmu_ha": 3,
        "smooth_radius_m": 15,
        "open_radius_m": 10,
        "close_radius_m": 10,
        "simplify_tol_m": 5,
        "simplify_buffer_m": 0,
        "export_target": "zip"
      }'
```

*Month range expansion*

```bash
curl -X POST http://localhost:8000/api/zones/production \
  -H "Content-Type: application/json" \
  -d '{
        "aoi_geojson": {"type": "Polygon", "coordinates": [[[149.7, -28.8], [149.8, -28.8], [149.8, -28.7], [149.7, -28.7], [149.7, -28.8]]]},
        "aoi_name": "Lot1_RP12345",
        "start_month": "2024-01",
        "end_month": "2024-04",
        "smooth_radius_m": 15,
        "open_radius_m": 10,
        "close_radius_m": 10,
        "simplify_tol_m": 5,
        "simplify_buffer_m": 0,
        "export_target": "gcs",
        "gcs_bucket": "zones-bucket",
        "gcs_prefix": "clients/demo"
      }'
```

When targeting Google Cloud Storage or Drive the endpoint requires the
destination details (`gcs_bucket` for `gcs`, an optional `gcs_prefix`, and Drive
will reuse the configured `GEE_DRIVE_FOLDER`).

The response summarises the generated artefacts and stability analysis:

```json
{
  "ok": true,
  "ym_start": "2024-03",
  "ym_end": "2024-05",
  "paths": {
    "raster": "zones/PROD_202403_202405_Lot1_RP12345_zones.tif",
    "vectors": "zones/PROD_202403_202405_Lot1_RP12345_zones.shp",
    "vector_components": {
      "shp": "zones/PROD_202403_202405_Lot1_RP12345_zones.shp",
      "shx": "zones/PROD_202403_202405_Lot1_RP12345_zones.shx",
      "dbf": "zones/PROD_202403_202405_Lot1_RP12345_zones.dbf",
      "cpg": "zones/PROD_202403_202405_Lot1_RP12345_zones.cpg",
      "prj": "zones/PROD_202403_202405_Lot1_RP12345_zones.prj"
    },
    "zonal_stats": "zones/PROD_202403_202405_Lot1_RP12345_zones_zonal_stats.csv"
  },
  "tasks": {
    "raster": {"id": "task_r", "state": "READY", "destination_uri": "gs://zones/demo.tif"},
    "vectors": {"id": "task_v", "state": "READY", "destination_uri": "gs://zones/demo.shp"}
  },
  "metadata": {
    "used_months": ["2024-03", "2024-04", "2024-05"],
    "skipped_months": ["2024-04"],
    "stability": {
      "initial_threshold": 0.25,
      "final_threshold": 0.6,
      "survival_ratio": 0.34,
      "surviving_pixels": 14250,
      "total_pixels": 42000,
      "low_confidence": false
    }
  },
  "debug": {
    "requested_months": ["2024-03", "2024-04", "2024-05"],
    "used_months": ["2024-03", "2024-05"],
    "skipped_months": ["2024-04"],
    "retry_thresholds": [0.25, 0.5, 0.6],
    "stability": {
      "thresholds_tested": [0.25, 0.5, 0.6, 0.8],
      "survival_ratios": [0.12, 0.22, 0.34, 0.38],
      "target_ratio": 0.2,
      "low_confidence": false
    }
  },
  "bucket": "zones-bucket",
  "prefix": "clients/demo/zones/PROD_202403_202405_Lot1_RP12345_zones"
}
```

The `debug` block mirrors the metadata exposed by the service and includes the
stability-mask retry thresholds and per-iteration survival ratios. When the
system has to relax the stability threshold to satisfy the minimum survival
ratio (`target_ratio`) the `low_confidence` flag is set; operators should treat
those exports as less reliable and consider collecting additional observations
for the period.

#### Masking controls

- `mask_mode`: `"strict" | "relaxed" | "adaptive"` (default adaptive). Adaptive tries stricter SCL
  filters first, then relaxes only when needed to reach `min_valid_ratio`.
- `min_valid_ratio`: coverage target (default `0.25`) used for per-month tier selection
  and for the pre-stability coverage guard.
- `stability_adaptive`: when true, the CV mask is only applied if coverage after
  stability remains ≥ `min_valid_ratio`.
- `cv_mask_threshold`: coefficient of variation threshold (`std/mean`); values in the
  `0.30–0.40` range retain more pixels.

### Zones Troubleshooting
- **E_NDVI_BAND**: Expected a single band named `NDVI`. Ensure the NDVI helper renames to `NDVI` and monthly composites select it.
- **E_MASK_SHAPE**: NDVI mask must be single-band. Use the intersection of `B8` and `B4` masks.
- **E_COVERAGE_LOW**: Too few valid pixels before stability masking. Inspect diagnostics for `per_month_preview`/`mask_tiers` and widen the month window if needed.
- **E_STABILITY_EMPTY**: CV mask removed too many pixels.
  - We now compute CV only where at least `min_obs_for_cv` months exist and clamp by tiny mean.
  - With `"stability_adaptive": true` (default), the pipeline bypasses stability if post-coverage < `min_valid_ratio`.
  - To force failure instead, set `"stability_enforce": true`.
- **Coverage**: To improve coverage, expand the month range, relax SCL filters to include classes 3/7, raise `cloud_prob_max` into the 60–70 range, or temporarily disable the stability mask to gauge impact.
- **E_RANGE_EMPTY**: NDVI_min == NDVI_max before classification. Fix by:
  - Ensuring NDVI uses B8/B4 float math (no visualize, no integer rounding).
  - Relaxing SCL/cloud masks (e.g., include classes 3/7) and/or widening months to include a greener + drier period.
  - Verifying AOI intersects Sentinel-2 coverage and reduction region uses `buffer(5).bounds(1)` with `scale=10, bestEffort, tileScale=4`.
- **E_BREAKS_COLLAPSED** (percentiles only): NDVI spread too small for distinct thresholds. Use `method=ndvi_kmeans` or widen the date range.

### Export package layout

All exports are staged under `OUTPUT_DIR/<job_id>` locally before being zipped
or uploaded. Production zone jobs create a `zones/` directory with the raster,
vector, and statistics artefacts. A representative archive looks like:

```
OUTPUT_DIR/
└── job-12345/
    ├── metadata.json
    ├── zones/
    │   ├── PROD_202403_202405_Lot1_RP12345_zones.tif
    │   ├── PROD_202403_202405_Lot1_RP12345_zones.shp
    │   ├── PROD_202403_202405_Lot1_RP12345_zones.shx
    │   ├── PROD_202403_202405_Lot1_RP12345_zones.dbf
    │   ├── PROD_202403_202405_Lot1_RP12345_zones.cpg
    │   ├── PROD_202403_202405_Lot1_RP12345_zones.prj
    │   └── PROD_202403_202405_Lot1_RP12345_zones_zonal_stats.csv  # present when include_zonal_stats=true
    └── zones_metadata.json
```

The PRJ file declares EPSG:4326 for compatibility with desktop GIS clients, and
the CSV contains optional zonal statistics (one row per zone with temporal NDVI
metrics). When `export_target=zip` the ZIP root will mirror this folder tree;
cloud targets write the same files to the destination bucket or Drive folder
using the `prefix` from the API response.

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

 
