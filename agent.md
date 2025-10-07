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
