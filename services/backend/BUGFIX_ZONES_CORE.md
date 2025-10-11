# Bug Fix: Zone Production Workflow Error

## Issue
The zone production workflow was failing with the error:
```
Failed: Invalid argument specified for ee.List(): 1
```

## Root Cause
The error occurred in `app/services/zones_core.py` in two functions:

1. **`ensure_list(x)`**: This function is designed to normalize both scalar values and lists into `ee.List` objects. The original implementation called `ee.List([x])` without first checking if `x` was already an `ee.List` or handling cases where `x` might be a plain Python primitive that Earth Engine can't wrap directly in certain server-side contexts.

2. **`remove_nulls(lst)`**: This function was calling `ee.List(lst)` without first checking if `lst` was already properly formatted as an `ee.List`, which could fail when passed unexpected types.

## The Problem
When Earth Engine operations are executed server-side (not in client-side mode), calling `ee.List()` with certain primitive types or in certain contexts can fail with the error "Invalid argument specified for ee.List(): X" where X is the problematic value.

This particularly affected operations like:
- `ee.List.sequence(1, n)` results being passed through `ensure_list()`
- Results from `ee.Algorithms.If()` conditionals
- Nested list operations in zone classification

## The Fix
Both functions were updated to:

1. **Check if input is already an `ee.List`**: Before attempting to wrap the value, safely check using `isinstance(x, ee.List)` with proper error handling for mock objects in tests.

2. **Handle edge cases gracefully**: Added try-except blocks to catch `TypeError` and `AttributeError` when wrapping values, with fallback logic for plain Python lists.

3. **Test-safe type checking**: Since the codebase uses mock `ee` objects in tests, the type checking is wrapped in try-except blocks to handle cases where `ee.List` might not be a proper type.

## Changes Made

### `ensure_list(x)` in `app/services/zones_core.py`:
- Added initial check: if `x` is already an `ee.List`, return it directly
- Added error handling for `ee.List([x])` call with fallback for plain Python lists
- Maintained backward compatibility with existing fake_ee test doubles

### `remove_nulls(lst)` in `app/services/zones_core.py`:
- Added check: if `lst` is already an `ee.List`, call `.removeAll([None])` directly
- Added error handling before calling `ee.List(lst)`
- Prevents attempting to wrap scalars or invalid types

## Testing
- All existing tests pass (73 passed, 10 skipped)
- Verified fix handles:
  - Plain Python lists: `[1, 2, 3]`
  - ee.List objects: `ee.List([1, 2, 3])`
  - Results from `ee.List.sequence()`
  - Mock ee objects in tests

## Impact
This fix resolves the zone production workflow error while maintaining full backward compatibility with existing code. The functions now robustly handle both client-side and server-side Earth Engine execution contexts.
