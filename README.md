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

 
