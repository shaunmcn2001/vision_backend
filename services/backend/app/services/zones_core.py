"""Shared helpers for zones-related Earth Engine list handling."""

import ee


def ensure_list(x):
    """
    EE-safe: return an ee.List whether x is scalar or already a list.
    - If x is ee.List: return x directly
    - If x is scalar (ee.Number/String/bool or Python primitive): wrap in ee.List
    """
    # Check if x is already an ee.List (safe check for both real and mock ee)
    try:
        if isinstance(ee.List, type) and isinstance(x, ee.List):
            return x
    except (AttributeError, TypeError):
        pass
    
    # Try wrapping in ee.List
    try:
        candidate = ee.List([x])
    except (TypeError, AttributeError) as e:
        # If ee.List([x]) fails (e.g., x is a plain Python int in server context),
        # return x wrapped as-is assuming it's already list-like
        if isinstance(x, list):
            return ee.List(x)
        # For other types, re-raise since we can't handle it
        raise
    except Exception as e:
        # Handle ee.EEException if ee module is available
        if hasattr(ee, 'EEException') and isinstance(e, ee.EEException):
            if isinstance(x, list):
                return ee.List(x)
        raise
    
    flatten = getattr(candidate, "flatten", None)
    if callable(flatten):
        try:
            return flatten()
        except Exception:  # pragma: no cover - fall back for fake EE doubles
            pass
    if isinstance(candidate, list):
        flattened = []
        for item in candidate:
            if isinstance(item, list):
                flattened.extend(item)
            else:
                flattened.append(item)
        return type(candidate)(flattened)
    return candidate


def remove_nulls(lst):
    """Remove nulls from an ee.List (Filter.notNull is for Collections, not ee.List)."""
    return ee.List(lst).removeAll([None])


def as_number(x):
    """Normalize any scalar-like expression to ee.Number (useful for If(...,1,0))."""
    return ee.Number(x)
