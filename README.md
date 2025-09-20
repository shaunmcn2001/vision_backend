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
- 2025-09-20: Introduced CRS heuristics for projection-less shapefiles, documented the new warnings, and expanded tests for Australian EPSG detection.
- 2025-09-20: Added EPSG:4326 fallback handling for shapefile uploads, surfaced CRS warnings to clients, and updated tests to cover projection defaults.
- 2025-09-20: Converted NDVI export regions to Web Mercator before requesting downloads and added regression coverage to guard CRS handling.
- 2025-09-20: Added repository-wide collaboration guidelines (`AGENTS.md`) and documented the expectation to log future changes here.
