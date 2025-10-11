"""Shared helpers for zones-related Earth Engine list handling."""

import ee


def ensure_list(x):
    """
    EE-safe: return an ee.List whether x is scalar or already a list.
    - If x is ee.List: ee.List([x]).flatten() -> x
    - If x is scalar (ee.Number/String/bool): -> [x]
    """
    candidate = ee.List([x])
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
