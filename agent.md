## Surgical edit policy
- Anchor-based edits only; abort if anchors missing.

## Diagnostics export
- The /zones/production endpoint accepts `diagnostics=true`.
- When enabled, service returns diagnostics in JSON and also writes `diagnostics/diagnostics.json` inside the export ZIP.
- Error codes: E_INPUT_AOI, E_NO_MONTHS, E_FIRST_MONTH_EMPTY, E_MEAN_EMPTY, E_MEAN_CONSTANT, E_STABILITY_EMPTY, E_BREAKS_COLLAPSED, E_FEW_CLASSES, E_VECT_EMPTY, E_EXPORT_FAIL_*.

### Zones Mean NDVI Guardrails
- NDVI must be computed with float math and named 'NDVI'.
- Monthly composite must return an image with band 'NDVI'.
- Cloud mask should prefer SCL; avoid over-aggressive QA masks.
- Mean NDVI min/max uses region=aoi.buffer(5).bounds(1), scale=10, bestEffort=True, tileScale=4.
- Guard stages:
  - pre_mean_valid_mask_sum: ensure non-zero valid pixels before mean().
  - ndvi_stats: require NDVI_min and NDVI_max present and NDVI_min < NDVI_max.
### Adaptive Monthly NDVI Policy
- Build per-month composites with tiered masks (strict SCL → relaxed SCL → prob → none).
- Accept the first variant meeting thresholds: VALID_RATIO ≥ 0.35 and NDVI_SPREAD ≥ 0.08.
- Server-side filter: keep only images with band 'NDVI' (`listContains('system:band_names','NDVI')`).
- Guard checks the first **valid** composite, not the first requested month.
- Diagnostics saved: MASK_TIER, VALID_RATIO, NDVI_SPREAD, NDVI_min/max, and first valid month (ym).

### Fix: Invalid argument specified for ee.List(): 1

**Problem:**  
The zones pipeline occasionally failed when scalars (e.g., `1`, `ee.Number`, or `If(...)` outputs) were passed to `ee.List()`.  

**Fix Summary:**  
- Added `safe_ee_list()` in `ee_utils.py` to wrap scalars safely.  
- Replaced all `ee.List(...)` calls in `zones_core.py` and `zones.py` with `safe_ee_list()`.  
- Added logging to track NDVI thresholds and list creation during runtime.  

**Testing:**  
✅ `pytest -q` passes  
✅ Zone generation runs without `Invalid argument specified for ee.List(): 1`
✅ Logs confirm all lists contain valid iterable values before being passed to Earth Engine.

## 2025-10-11 – Fix: 'Invalid argument specified for ee.List(): 1' in zones
- Root cause: Scalars (1/0/If results) being passed to ee.List().
- Fixes:
  - Added EE-safe helpers ensure_list/ensure_number/cat_one/remove_nulls.
  - Stability replicated from GEE JS: thresholds mapped to images, combined via ImageCollection.max().
  - Normalized reducer bands (NDVI_mean → NDVI) before k-means.
- Tests:
  - pytest -q passes locally.
  - Zones run completes without ee.List() argument errors.

## 2025-10-11 – Remove legacy GEE zones creation; PyQGIS only
- **Removed old GEE-based zones vectorization**: Eliminated percentile thresholds + GEE polygonization (reduceToVectors).
- **Zones are now created with PyQGIS** from the NDVI GeoTIFF: k-means/quantiles → polygonize → MMU → smooth/simplify → export.
- **API fields removed**: `method` (legacy), `palette`/`thresholds` tied to GEE vectorization outputs.
- **API fields kept/added**: `n_classes`, `classifier` ("kmeans" or "quantiles"), `mmu_ha`, `smooth_radius_m`, `simplify_tol_m`, `export_vector_format` ("gpkg", "geojson", "shp").
- **Response now includes**: `paths.zones_vector` (gpkg/geojson/shp path), `metadata.zones_pyqgis` (PyQGIS metadata).
- **NDVI production (Earth Engine)** remains unchanged; only vectorization moved to PyQGIS.
- **Removed functions**: `_prepare_vectors`, `_dissolve_vectors`, `_simplify_vectors`, `_build_percentile_zones`, `_classify_by_percentiles`.
- **Runtime dependency**: QGIS/GDAL must be available on the server/container (QGIS_PREFIX_PATH etc.). PyQGIS is not pip-installable; it relies on system-installed QGIS.
- **Tests**: Updated to remove assertions for legacy `palette`/`percentile_thresholds`. Added `test_zones_pyqgis.py` for PyQGIS flow (requires QGIS or mocking).
