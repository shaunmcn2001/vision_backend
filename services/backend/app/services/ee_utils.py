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
    """Always return an ee.List, flattening nested lists."""

    return ee.List([x]).flatten()


def ensure_number(x):
    """Convert x to ee.Number safely."""

    try:
        return ee.Number(x)
    except Exception:  # pragma: no cover - keep legacy default behaviour
        return ee.Number(0)


def remove_nulls(lst):
    """Remove nulls from ee.List or iterable."""

    try:
        return ee.List(lst).removeAll([None])
    except Exception:  # pragma: no cover - maintain compatibility
        return ee.List([])


def cat_one(lst, val):
    """Append val to lst safely."""

    try:
        return ee.List(lst).cat(safe_ee_list(val))
    except Exception:  # pragma: no cover - defensive fallback
        return ee.List(lst)
