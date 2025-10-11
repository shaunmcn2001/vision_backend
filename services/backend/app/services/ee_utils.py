from __future__ import annotations

import collections.abc
import logging

import ee


def safe_ee_list(val):
    """
    Wraps any scalar or unknown iterable safely as ee.List.
    Prevents Invalid argument for ee.List(): 1 errors.
    """

    try:
        try:
            is_ee_list = isinstance(val, ee.List)
        except TypeError:  # pragma: no cover - fake EE doubles may use callables
            is_ee_list = False
        if is_ee_list:
            return val
        if isinstance(val, collections.abc.Iterable) and not isinstance(
            val, (str, bytes)
        ):
            return ee.List(val)
        return ee.List([val])
    except Exception as exc:  # pragma: no cover - defensive guard
        logging.warning(
            "safe_ee_list: could not wrap %r (%r), falling back to [val]",
            val,
            exc,
        )
        return ee.List([val])


def ensure_list(x):
    """
    EE-safe: returns an ee.List whether x is scalar or already a list.
    Implementation: wrap-then-flatten.
    - If x is ee.List -> [x].flatten() == x
    - If x is scalar (ee.Number/String/bool/int/float) -> [x]
    """

    wrapped = ee.List([x])
    flatten = getattr(wrapped, "flatten", None)
    if callable(flatten):
        return flatten()
    if isinstance(x, (list, tuple)):
        return ee.List(x)
    return wrapped


def ensure_number(x):
    """Normalize expression/scalar to ee.Number."""

    return ee.Number(x)


def remove_nulls(lst):
    """Remove nulls from an ee.List (Filter.notNull is for collections)."""

    return ee.List(lst).removeAll([None])


def cat_one(lst, value):
    """Append a single value (scalar or list) to an ee.List safely."""

    return ee.List(lst).cat(ensure_list(value))
