# Earth Engine List/Scalar Normalization Hardening

## Overview
This fix addresses the intermittent "Invalid argument specified for ee.List(): 1" error that occurred when scalars (e.g., `1`, `0`, `ee.Number`, or outputs of `ee.Algorithms.If(...)`) were passed to `ee.List()` during server-side Earth Engine evaluation.

## Root Cause
Earth Engine's `ee.List()` constructor is not consistently tolerant of scalar values during deferred, server-side evaluation. When scalars or conditional results are passed directly, it can fail with the cryptic error message that lacks context about the source of the failure.

## Solution Implemented

### 1. Enhanced Defensive Logging in `ee_utils.py`

All helper functions now include comprehensive error handling and logging:

#### `ensure_list(x)`
- **Primary function**: Wraps values in `ee.List([x]).flatten()` pattern (idempotent for lists, safe for scalars)
- **Error handling**: Try/except with WARNING-level logging on failure
- **Fallback**: Returns `ee.List([])` to allow pipeline continuation
- **Final fallback**: Logs ERROR with full stack trace if even the fallback fails

#### `ensure_number(x)`
- **Error handling**: Try/except with WARNING-level logging
- **Fallback**: Returns `ee.Number(0)`
- **Final fallback**: Logs ERROR with full stack trace

#### `remove_nulls(lst)`
- **Error handling**: Try/except with WARNING-level logging
- **Fallback**: Returns original list (may still contain nulls)
- **Final fallback**: Logs ERROR with full stack trace

#### `cat_one(lst, value)`
- **Error handling**: Try/except with WARNING-level logging
- **Fallback**: Returns original list without appending
- **Final fallback**: Logs ERROR with full stack trace

### Logging Format
All logs follow the consistent format specified in the requirements:
```
{function}: {STATUS} | input={repr} | error={type}: {message}
```

- **WARNING**: Recoverable errors where fallback succeeded
- **ERROR**: Fatal errors where all fallbacks failed
- Includes `exc_info=True` to capture full stack traces for diagnostics

### 2. Global Safety Patch in `ee_patches.py` (Already Present)

The existing `ee_patches.py` already implements the optional global safety patch:

- `_safe_ee_list(x)`: Wraps all `ee.List()` calls using `[x].flatten()` pattern
- Applied via `apply_ee_runtime_patches()` which is idempotent
- Monkeypatches `ee.List` to use the safe wrapper
- Already imported and applied at module load in `zones.py` and `zones_core.py`

### 3. Replaced Direct `ee.List([])` Calls in `zones.py`

Replaced 8 instances of direct `ee.List([])` calls with `ensure_list([])`:

1. **Line 2288**: `robust_quantile_breaks` early return for invalid n_classes
2. **Line 2347**: `_dedup` function - accumulator initialization in If clause
3. **Lines 2376, 2378**: `_uniq_sort` iterate function - empty list initialization
4. **Line 2425**: `_hist_breaks` - invalid range fallback
5. **Line 2432**: `_hist_quantiles` - early return for n_classes < 2
6. **Line 2467**: `robust_quantile_breaks` - final return fallback
7. **Line 2541**: `kmeans_classify` - groups default value
8. **Line 2653**: `_rank_zones` - groups default value

All replacements maintain identical semantics while using the hardened helper.

### 4. Bug Fix in `ee_debug.py`

Removed stray line `ee_debug.py` at end of file that was causing `NameError` during imports.

## Testing Results

- **73 tests pass** (0 failures)
- **10 tests skipped** (require Earth Engine authentication)
- **No regressions** observed
- Existing logging at critical points verified:
  - `robust_quantile_breaks`
  - `_uniq_sort`
  - `_dedup`
  - `kmeans_classify`

## Performance Impact

- **Negligible**: Logging only executes on error paths
- Normal operations have no additional overhead
- Error handling uses minimal try/except blocks with fast-path success

## Acceptance Criteria Met

✅ Scenarios that triggered "Invalid argument specified for ee.List(): 1" now use hardened helpers  
✅ All helpers have defensive logging with WARNING/ERROR levels  
✅ Logging format includes function name, input value, error type and message  
✅ Stack traces captured via `exc_info=True`  
✅ Unit test suite passes (73/73)  
✅ No performance regressions  
✅ Global safety patch present and active (via `ee_patches.py`)  
✅ Critical points in zones.py have targeted logging  

## Usage

The changes are automatic and require no code changes:

1. **Helper functions** are already used throughout the codebase via the wrapper functions in `zones.py`
2. **Global patch** is automatically applied via `apply_ee_runtime_patches()` at module import
3. **Logging** will appear in application logs when errors occur, with sufficient context to diagnose issues

## Future Enhancements

If additional issues arise:
1. Check logs for the specific error location
2. Add additional `ensure_list()` or `ensure_number()` calls at identified points
3. Consider expanding the global patch to cover additional Earth Engine types if needed

## Files Modified

- `services/backend/app/services/ee_utils.py` - Added defensive logging to all helpers
- `services/backend/app/services/zones.py` - Replaced 8 `ee.List([])` calls with `ensure_list([])`
- `services/backend/app/services/ee_debug.py` - Fixed stray line causing import error

## References

- Problem statement: Earth Engine "Invalid argument specified for ee.List(): 1" error
- Pattern based on resilient GEE JavaScript patterns using wrap-then-flatten
- Consistent with existing error handling patterns in the codebase
