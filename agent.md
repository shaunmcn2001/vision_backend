## Surgical edit policy
- Anchor-based edits only; abort if anchors missing.

## Diagnostics export
- The /zones/production endpoint accepts `diagnostics=true`.
- When enabled, service returns diagnostics in JSON and also writes `diagnostics/diagnostics.json` inside the export ZIP.
- Error codes: E_INPUT_AOI, E_NO_MONTHS, E_FIRST_MONTH_EMPTY, E_MEAN_EMPTY, E_MEAN_CONSTANT, E_STABILITY_EMPTY, E_BREAKS_COLLAPSED, E_FEW_CLASSES, E_VECT_EMPTY, E_EXPORT_FAIL_*.
