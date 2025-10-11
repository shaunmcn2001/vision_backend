from __future__ import annotations
import logging
import ee

logger = logging.getLogger(__name__)


def ensure_list(x):
    """
    Always return an ee.List safely.
    Wrap-then-flatten: if x is a list -> returns it; if scalar -> [x].
    """
    try:
        if isinstance(x, (list, tuple)):
            return ee.List(x)
        candidate = ee.List([x])
        flatten = getattr(candidate, "flatten", None)
        if callable(flatten):
            return flatten()
        return candidate
    except Exception as exc:
        logger.warning(
            "ensure_list: FAILED | input=%r | error=%s: %s",
            x,
            type(exc).__name__,
            str(exc),
            exc_info=True,
        )
        # Fallback: return empty list to allow pipeline to continue
        try:
            return ee.List([])
        except Exception as fallback_exc:
            logger.error(
                "ensure_list: ALL FALLBACKS FAILED | error=%s: %s",
                type(fallback_exc).__name__,
                str(fallback_exc),
                exc_info=True,
            )
            raise


def ensure_number(x):
    """Return ee.Number for any scalar-like EE expression (incl. If(...))."""
    try:
        return ee.Number(x)
    except Exception as exc:
        logger.warning(
            "ensure_number: FAILED | input=%r | error=%s: %s",
            x,
            type(exc).__name__,
            str(exc),
            exc_info=True,
        )
        # Fallback: return 0
        try:
            return ee.Number(0)
        except Exception as fallback_exc:
            logger.error(
                "ensure_number: ALL FALLBACKS FAILED | error=%s: %s",
                type(fallback_exc).__name__,
                str(fallback_exc),
                exc_info=True,
            )
            raise


def remove_nulls(lst):
    """Remove None from ee.List (Filter.notNull is for Collections, not Lists)."""
    try:
        return ee.List(lst).removeAll([None])
    except Exception as exc:
        logger.warning(
            "remove_nulls: FAILED | input=%r | error=%s: %s",
            lst,
            type(exc).__name__,
            str(exc),
            exc_info=True,
        )
        # Fallback: return the input as ee.List (may still have nulls)
        try:
            return ee.List(lst)
        except Exception as fallback_exc:
            logger.error(
                "remove_nulls: ALL FALLBACKS FAILED | error=%s: %s",
                type(fallback_exc).__name__,
                str(fallback_exc),
                exc_info=True,
            )
            raise


def cat_one(lst, value):
    """Append a single value (scalar or list) to ee.List safely."""
    try:
        return ee.List(lst).cat(ensure_list(value))
    except Exception as exc:
        logger.warning(
            "cat_one: FAILED | lst=%r, value=%r | error=%s: %s",
            lst,
            value,
            type(exc).__name__,
            str(exc),
            exc_info=True,
        )
        # Fallback: return the original list
        try:
            return ee.List(lst)
        except Exception as fallback_exc:
            logger.error(
                "cat_one: ALL FALLBACKS FAILED | error=%s: %s",
                type(fallback_exc).__name__,
                str(fallback_exc),
                exc_info=True,
            )
            raise
