"""Shared helpers for zones-related Earth Engine list handling."""

from __future__ import annotations

from typing import Any

import ee


def ensure_list(value, ee_api: Any | None = None):
    """Wrap scalars or nested lists in an ``ee.List`` and flatten the result."""
    api = ee_api or ee
    candidate = api.List([value])
    flatten = getattr(candidate, "flatten", None)
    if callable(flatten):
        try:
            return flatten()
        except Exception:  # pragma: no cover - defensive fallback
            pass
    if isinstance(candidate, list):
        flattened: list[Any] = []
        for item in candidate:
            if isinstance(item, list):
                flattened.extend(item)
            else:
                flattened.append(item)
        return type(candidate)(flattened)
    return candidate


def remove_nulls(lst, ee_api: Any | None = None):
    """Remove ``None`` values from an ``ee.List`` (or coercible input)."""
    api = ee_api or ee
    candidate = api.List(lst)
    remove_all = getattr(candidate, "removeAll", None)
    if callable(remove_all):
        try:
            return remove_all([None])
        except Exception:  # pragma: no cover - defensive fallback
            pass
    if isinstance(candidate, list):
        filtered = [item for item in candidate if item is not None]
        return type(candidate)(filtered)
    return candidate
