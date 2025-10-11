from __future__ import annotations
import ee


def ensure_list(x):
    """
    Always return an ee.List safely.
    Wrap-then-flatten: if x is a list -> returns it; if scalar -> [x].
    """
    if isinstance(x, (list, tuple)):
        return ee.List(x)
    candidate = ee.List([x])
    flatten = getattr(candidate, "flatten", None)
    if callable(flatten):
        return flatten()
    return candidate


def ensure_number(x):
    """Return ee.Number for any scalar-like EE expression (incl. If(...))."""
    return ee.Number(x)


def remove_nulls(lst):
    """Remove None from ee.List (Filter.notNull is for Collections, not Lists)."""
    return ee.List(lst).removeAll([None])


def cat_one(lst, value):
    """Append a single value (scalar or list) to ee.List safely."""
    return ee.List(lst).cat(ensure_list(value))
