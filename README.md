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

This payload succeeds without specifying `collection`, `scale`, or `crs`, relying on the defaults.
Behind the scenes the service now samples using a Web Mercator (`EPSG:3857`) projection so the 10 m scale reflects true metre-based averaging.


## NDVI GeoTIFF exports

The `/api/export` endpoint now resamples the NDVI band with bilinear interpolation at 10 m
resolution before clipping exports to the submitted geometry. This avoids nearest-neighbour
artifacts when comparing the GeoTIFFs against vector boundaries.


## Updates
- 2025-09-25: Replaced the shapefile CRS fallback with an explicit HTTP 400 requiring projection metadata and updated the shapefile utility tests.
  Ran `pytest services/backend/tests/test_shapefile_utils.py -q` (pass), while `ruff check .` / `ruff format --check .` still surface longstanding import/formatting violations, and `mypy services/backend` continues to fail because core dependencies and legacy modules lack typing support.
- 2025-09-24: Defaulted shapefile uploads without CRS metadata to EPSG:4326 with client-facing warnings, updated shapefile util tests for the new fallback, ran `pytest services/backend/tests/test_shapefile_utils.py -q` (pass), noted that `ruff check .` and `ruff format --check .` still flag longstanding import/formatting issues in legacy modules, and `mypy services/backend` continues to fail because project dependencies like shapely/pyshp/google-cloud lack bundled typing stubs.
- 2025-09-23: Added CRS detector hints for projection-less shapefile uploads, appended the human-readable guidance to HTTP 400 responses and logs, extended tests to assert the hint surfaces when metadata is missing, ran `pytest services/backend/tests -q` (pass), and `ruff check .` / `ruff format --check .` continue to report longstanding import-grouping and formatting violations in legacy modules.
- 2025-09-22: Required shapefile uploads without projection metadata to supply a .prj or source_epsg, refreshed heuristic warnings to
  direct clients toward sharing CRS details, documented the new HTTP 400 expectation in tests, ran `pytest` (pass), and noted that
  `ruff check .` / `ruff format --check .` still flag pre-existing import and formatting issues.
- 2025-09-21: Forced Earth Engine geometries to use planar WGS84 (geodesic disabled) across exports, NDVI services, and tiles so shapefile footprints stay aligned, refreshed tests to assert the new kwargs, and ran `pytest`.
- 2025-09-20: Introduced CRS heuristics for projection-less shapefiles, documented the new warnings, and expanded tests for Australian EPSG detection.
- 2025-09-20: Added EPSG:4326 fallback handling for shapefile uploads, surfaced CRS warnings to clients, and updated tests to cover projection defaults.
- 2025-09-20: Converted NDVI export regions to Web Mercator before requesting downloads and added regression coverage to guard CRS handling.
- 2025-09-20: Added repository-wide collaboration guidelines (`AGENTS.md`) and documented the expectation to log future changes here.
