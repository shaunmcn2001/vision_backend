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
## 2025-10-11 – Hotfix: EE list/collection safety + NDVI normalization + stability mask
- Added helpers: ensure_list/ensure_number/remove_nulls/cat_one (ee_utils.py).
- Normalized NDVI reducer outputs so classifier always selects a single band 'NDVI' (ndvi_helpers.py).
- Rewrote stability mask builder to return Images on all branches; safe ImageCollection.fromImages usage (stability_mask.py).
- Applied mechanical fixes: replaced risky ee.List(...) casts, ensured .map(...) returns Images, and added optional input guards.
- Purpose: eliminate "Invalid argument specified for ee.List(): 1" and mixed-type ImageCollection errors without changing business logic.

## 2025-10-11 – Earth Engine debug tracing
- Added ee_debug.py helper and runtime hook to print file, line, and source for all EE errors.
- Automatically activated via ee_patches global hook.
- Example output:
    [EE TRACE] Invalid argument specified for ee.List(): 1
      at zones_core.py:421 (compute_breaks)
      line: l2 = ee.List(l).sort()
- Use @debug_wrap to decorate risky EE functions for localized tracing.
