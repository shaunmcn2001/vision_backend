import ee
import logging

logger = logging.getLogger(__name__)


def ensure_list(x):
    """
    EE-safe: returns an ee.List whether x is scalar or already a list.
    - If x is ee.List: flatten returns the list.
    - If x is scalar (ee.Number/String/bool/int/float), wraps in a single-element list.
    
    This function handles the "Invalid argument specified for ee.List()" error
    that can occur when Earth Engine operations are executed server-side.
    """
    # Check if x is already an ee.List - if so, return it directly
    try:
        if isinstance(x, ee.List):
            return x
    except (TypeError, AttributeError):
        # Handle cases where ee.List might be a mock in tests
        pass
    
    try:
        return ee.List(x)
    except Exception as e:
        logger.warning(
            "ensure_list: ee.List(x) failed with %s: %s at %s, trying fallback",
            type(e).__name__,
            str(e),
            f"type={type(x).__name__}"
        )
        try:
            candidate = ee.List([x])
            flatten = getattr(candidate, "flatten", None)
            if callable(flatten):
                try:
                    return flatten()
                except Exception:  # pragma: no cover - fake EE fallback
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
        except Exception as fallback_err:
            logger.error(
                "ensure_list: ALL FALLBACKS FAILED at line=%s | input type=%s, value=%s, error=%s: %s",
                "ee_utils.py:ensure_list",
                type(x).__name__,
                str(x)[:100],
                type(fallback_err).__name__,
                str(fallback_err),
                exc_info=True
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
    # Check if lst is already an ee.List - if so, use it directly
    try:
        if isinstance(lst, ee.List):
            return lst.removeAll([None])
    except (TypeError, AttributeError):
        # Handle cases where ee.List might be a mock in tests
        pass
    
    try:
        return ee.List(lst).removeAll([None])
    except Exception as e:
        logger.warning(
            "remove_nulls: ee.List(lst) failed at %s | type=%s, error=%s: %s, trying fallback",
            "ee_utils.py:remove_nulls",
            type(lst).__name__,
            type(e).__name__,
            str(e)
        )
        # Try to handle edge cases
        try:
            # If it's a scalar, wrap it first
            wrapped = ensure_list(lst)
            return wrapped.removeAll([None])
        except Exception as fallback_err:
            logger.error(
                "remove_nulls: ALL FALLBACKS FAILED at line=%s | input type=%s, error=%s: %s",
                "ee_utils.py:remove_nulls",
                type(lst).__name__,
                type(fallback_err).__name__,
                str(fallback_err),
                exc_info=True
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
