import collections.abc
import ee
import logging

logger = logging.getLogger(__name__)


def _is_iterable(value: object) -> bool:
    try:
        return isinstance(value, collections.abc.Iterable) and not isinstance(
            value, (str, bytes)
        )
    except Exception:  # pragma: no cover - defensive against odd mocks
        return False


def ensure_list(x):
    """Return an :class:`ee.List` for scalars, iterables, or existing lists."""
    try:
        if isinstance(x, ee.List):
            return x
    except (TypeError, AttributeError):  # pragma: no cover - fake EE doubles
        pass

    if _is_iterable(x):
        try:
            return ee.List(x)
        except Exception as exc:
            logger.warning(
                "ensure_list: direct ee.List conversion failed | type=%s error=%s: %s",  # noqa: E501
                type(x).__name__,
                type(exc).__name__,
                str(exc),
            )

    try:
        return ee.List([x])
    except Exception as fallback_err:
        logger.error(
            "ensure_list: fallback wrapping failed | type=%s error=%s: %s",  # noqa: E501
            type(x).__name__,
            type(fallback_err).__name__,
            str(fallback_err),
            exc_info=True,
        )
        raise


def ensure_number(x):
    """
    Normalize any scalar-like expression to ee.Number.
    Useful when condensing boolean/if results into numbers.
    """
    try:
        return ee.Number(x)
    except Exception as e:
        logger.error(
            "ensure_number: FAILED at line=%s | type=%s, error=%s: %s",
            "ee_utils.py:ensure_number",
            type(x).__name__,
            type(e).__name__,
            str(e),
            exc_info=True
        )
        raise


def remove_nulls(lst):
    """
    Remove nulls from an ee.List.
    .filter(ee.Filter.notNull(...)) is for collections, not lists.

    This function handles the "Invalid argument specified for ee.List()" error
    that can occur when attempting to wrap scalars or invalid types.
    """
    try:
        if isinstance(lst, ee.List):
            return lst.removeAll([None])
    except (TypeError, AttributeError):  # pragma: no cover - fake EE doubles
        pass

    try:
        candidate = ee.List(lst)
    except Exception as exc:
        logger.warning(
            "remove_nulls: direct ee.List conversion failed | type=%s error=%s: %s",  # noqa: E501
            type(lst).__name__,
            type(exc).__name__,
            str(exc),
        )
        candidate = ensure_list(lst)

    try:
        return candidate.removeAll([None])
    except Exception as fallback_err:
        logger.error(
            "remove_nulls: removeAll failed | type=%s error=%s: %s",  # noqa: E501
            type(lst).__name__,
            type(fallback_err).__name__,
            str(fallback_err),
            exc_info=True,
        )
        raise


def cat_one(lst, value):
    """
    Append a single value (scalar or list) to an ee.List safely.
    Converts scalars to a one-element list via ensure_list() before concatenation.
    """
    try:
        return ee.List(lst).cat(ensure_list(value))
    except Exception as e:
        logger.error(
            "cat_one: FAILED at line=%s | lst type=%s, value type=%s, error=%s: %s",
            "ee_utils.py:cat_one",
            type(lst).__name__,
            type(value).__name__,
            type(e).__name__,
            str(e),
            exc_info=True
        )
        raise
