# Error Logging Enhancement for Earth Engine Operations

## Overview
This document describes the comprehensive error logging added to diagnose and fix the "Invalid argument specified for ee.List(): 1" error that occurs during production zones building.

## Problem Statement
Users were experiencing the error "Invalid argument specified for ee.List(): 1" during the production zones building phase, with no information about where in the code the error was occurring or what was causing it.

## Root Cause
The error occurs when Earth Engine operations are executed server-side and certain primitive types or results from operations like `ee.Algorithms.If()` are passed to `ee.List()` without proper type checking.

## Solution Implemented

### 1. Enhanced `app/services/ee_utils.py`

All four utility functions now have comprehensive error logging:

#### `ensure_list(x)`
- **Check before conversion**: Tests if input is already an `ee.List` before attempting conversion
- **Warning-level logging**: When the first `ee.List(x)` call fails, logs the error type, message, and input type
- **Error-level logging**: When all fallback mechanisms fail, logs complete details including:
  - Function location: `ee_utils.py:ensure_list`
  - Input type
  - Input value (first 100 chars)
  - Error type and message
  - Full stack trace via `exc_info=True`

#### `remove_nulls(lst)`
- **Check before conversion**: Tests if input is already an `ee.List` before attempting conversion
- **Warning-level logging**: When initial conversion fails
- **Fallback through ensure_list**: If direct conversion fails, attempts to use `ensure_list()` first
- **Error-level logging**: When all fallbacks fail, includes function location and full context

#### `ensure_number(x)`
- **Error-level logging**: On any failure, logs:
  - Function location: `ee_utils.py:ensure_number`
  - Input type
  - Error type and message
  - Full stack trace

#### `cat_one(lst, value)`
- **Error-level logging**: On any failure, logs:
  - Function location: `ee_utils.py:cat_one`
  - List type and value type
  - Error type and message
  - Full stack trace

### 2. Enhanced `app/services/zones.py`

Added strategic error logging at critical points in the zone production workflow:

#### `robust_quantile_breaks()`
- Logs function entry with parameters
- Logs success/failure of percentile mapping
- Logs completion of `_uniq_sort` operation
- Catches and logs errors with full context

#### `_uniq_sort()` inner function
- Logs completion of sort and null removal
- Logs completion of iterate operation
- Catches errors in the complex deduplication logic

#### `_dedup()` inner function
- Wraps the complex `ee.Algorithms.If()` logic in try-except
- Logs detailed type information for both `idx` and `prev` parameters
- Helps identify which specific iteration causes problems

#### `kmeans_classify()`
- Logs completion of group means computation
- Logs completion of remap list building
- Logs completion of relabeling
- Catches errors at each critical step

## Log Message Format

Error messages follow this consistent format:
```
{function_name}: {STATUS} at line={location} | {context_details} | error={error_type}: {error_message}
```

Example:
```
ensure_list: ALL FALLBACKS FAILED at line=ee_utils.py:ensure_list | input type=ComputedObject, value=ComputedObject(...), error=TypeError: Invalid argument specified for ee.List(): 1
```

## How to Use These Logs

### In Production

1. **Enable appropriate log level**: Set logging to WARNING or DEBUG level to capture the messages
   ```python
   logging.basicConfig(level=logging.WARNING)
   ```

2. **Search for error markers**: Look for:
   - `FAILED` - indicates a critical error
   - `ALL FALLBACKS FAILED` - indicates all recovery attempts exhausted
   - Function names: `ensure_list`, `remove_nulls`, `ensure_number`, `cat_one`
   - Zone functions: `robust_quantile_breaks`, `kmeans_classify`

3. **Identify the exact failure point**: The log message will show:
   - Which function failed
   - What type of input caused the failure
   - The error message from Earth Engine
   - Line/location information

### Example Diagnostic Workflow

If you see:
```
WARNING - app.services.ee_utils - ensure_list: ee.List(x) failed with TypeError: Invalid argument specified for ee.List(): 1 at type=int, trying fallback
```

This tells you:
- The `ensure_list` function received an integer value `1`
- Earth Engine rejected this value directly
- The function is attempting fallback recovery

If followed by:
```
ERROR - app.services.zones - _dedup: FAILED at line=zones.py:_dedup | idx type=ComputedObject, prev type=ComputedObject, error=TypeError: ...
```

This tells you:
- The error occurred inside the `_dedup` inner function
- Both `idx` and `prev` were EE ComputedObject types
- The error happened during the zone classification deduplication logic

## Testing

All existing tests pass (73 passed, 10 skipped) with the new logging in place.

A test script at `/tmp/test_logging.py` validates the logging behavior with mock Earth Engine objects.

## Benefits

1. **Precise error location**: Know exactly which function and line caused the error
2. **Type information**: See what types of values were being processed
3. **Context preservation**: Full stack traces available via `exc_info=True`
4. **Debugging capability**: Warning-level logs show recovery attempts before final failure
5. **Production-ready**: Logging levels are appropriate (WARNING for recoverable, ERROR for fatal)

## Performance Impact

Minimal - logging only occurs:
- At DEBUG level for successful operations (can be disabled in production)
- At WARNING level when fallbacks are triggered (rare)
- At ERROR level when operations fail (exceptional case)

The type checking (`isinstance`) has negligible overhead and is wrapped in try-except to handle mock objects in tests.
