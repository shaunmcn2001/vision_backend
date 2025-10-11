"""Client-side Earth Engine utility functions for type checking and conversion.

This module provides utilities for handling Earth Engine objects and Python primitives
without creating server-side ComputedObjects that can't be wrapped. All type checking
is performed client-side using Python isinstance checks.
"""

from __future__ import annotations

import logging
from typing import Any

import ee

logger = logging.getLogger(__name__)


def _is_ee_object(value: Any) -> bool:
    """Check if a value is an Earth Engine object.
    
    Args:
        value: Any value to check
        
    Returns:
        True if value is an ee object, False otherwise
    """
    if value is None:
        return False
    try:
        module = getattr(value.__class__, "__module__", "")
        if module.startswith("ee.") or module == "ee":
            return True
        # Check if it's an instance of ee.ComputedObject
        if hasattr(ee, "ComputedObject") and isinstance(value, ee.ComputedObject):
            return True
    except (AttributeError, TypeError):
        pass
    return False


def ensure_list(value: Any) -> ee.List:
    """Ensure a value is an ee.List using client-side type checking.
    
    Uses Python isinstance checks to determine if the input is already an ee.List.
    If it is, returns it directly. Otherwise, wraps the value in an ee.List.
    This avoids creating server-side ComputedObjects from ee.Algorithms.If().
    
    Args:
        value: A Python primitive, ee object, or ee.List to ensure is a list
        
    Returns:
        An ee.List containing the value
        
    Examples:
        >>> ensure_list(5)  # Returns ee.List([5])
        >>> ensure_list(ee.List([1, 2, 3]))  # Returns the list as-is
        >>> ensure_list(ee.Number(5))  # Returns ee.List([ee.Number(5)])
    """
    # Client-side check: if it's already an ee.List, return it
    # Use try-except to handle cases where ee.List might not be a proper type (e.g., in tests)
    try:
        if isinstance(value, ee.List):
            return value
    except TypeError:
        # ee.List is not a proper type (might be mocked), check by class name
        if value.__class__.__name__ == 'List' and hasattr(value, 'cat'):
            return value
    
    # For Python lists, convert directly
    if isinstance(value, (list, tuple)):
        return ee.List(value)
    
    # For all other values (primitives, ee objects, ComputedObjects), wrap in a list
    try:
        return ee.List([value])
    except (TypeError, AttributeError) as e:
        value_type = type(value).__name__
        value_str = str(value)[:100]
        logger.warning(
            "ensure_list: ee.List([x]) failed with %s: %s at type=%s, trying fallback",
            type(e).__name__,
            str(e),
            value_type
        )
        # Try alternative: if it's a ComputedObject, try wrapping differently
        try:
            # Last resort: try to create a list with the value directly passed to ee.List
            if hasattr(value, 'getInfo'):
                # It's an EE object, try to wrap it in a list
                return ee.List([value])
            # If it's a plain primitive that failed, log and re-raise
            raise
        except Exception as fallback_error:
            logger.error(
                "ensure_list: ALL FALLBACKS FAILED at line=ee_utils.py:ensure_list | "
                "input type=%s, value=%s, error=%s: %s",
                value_type,
                value_str,
                type(fallback_error).__name__,
                str(fallback_error),
                exc_info=True
            )
            raise


def ensure_number(value: Any) -> ee.Number | int | float:
    """Ensure a value is an ee.Number using client-side type checking.
    
    Uses Python isinstance checks to determine the type. If the value is already
    an ee.Number, returns it directly. For Python primitives, returns them as-is
    (ee functions handle automatic conversion). For other ee objects, wraps in ee.Number.
    This avoids creating server-side ComputedObjects from ee.Algorithms.If().
    
    Args:
        value: A Python number, ee.Number, or other value to ensure is a number
        
    Returns:
        An ee.Number or Python number
        
    Examples:
        >>> ensure_number(5)  # Returns 5 (Python int)
        >>> ensure_number(ee.Number(5))  # Returns the ee.Number as-is
        >>> ensure_number(ee.String("5"))  # Returns ee.Number(value)
    """
    # Client-side check: if it's already an ee.Number, return it
    # Use try-except to handle cases where ee.Number might not be a proper type (e.g., in tests)
    try:
        if isinstance(value, ee.Number):
            return value
    except TypeError:
        # ee.Number is not a proper type (might be mocked), check by class name
        if value.__class__.__name__ == 'Number' and hasattr(value, 'getInfo'):
            return value
    
    # For Python numbers, return as-is (ee functions handle conversion)
    if isinstance(value, (int, float)):
        return value
    
    # For other ee objects or ComputedObjects, wrap in ee.Number
    try:
        return ee.Number(value)
    except Exception as e:
        value_type = type(value).__name__
        value_str = str(value)[:100]
        logger.error(
            "ensure_number: FAILED at line=ee_utils.py:ensure_number | "
            "input type=%s, value=%s, error=%s: %s",
            value_type,
            value_str,
            type(e).__name__,
            str(e),
            exc_info=True
        )
        raise


def cat_one(lst: Any, value: Any) -> ee.List:
    """Concatenate a single value to a list using client-side operations.
    
    Uses Python list concatenation when possible, falling back to ee.List operations.
    This avoids unnecessary server-side operations.
    
    Args:
        lst: An ee.List, Python list, or value that can be converted to a list
        value: A value to append to the list
        
    Returns:
        An ee.List with the value appended
        
    Examples:
        >>> cat_one([], 1)  # Returns ee.List([1])
        >>> cat_one(ee.List([1, 2]), 3)  # Returns ee.List([1, 2, 3])
    """
    # Ensure lst is an ee.List
    is_ee_list = False
    try:
        is_ee_list = isinstance(lst, ee.List)
    except TypeError:
        # ee.List is not a proper type (might be mocked), check by class name
        is_ee_list = lst.__class__.__name__ == 'List' and hasattr(lst, 'cat')
    
    if not is_ee_list:
        lst = ee.List(lst) if isinstance(lst, (list, tuple)) else ee.List([lst])
    
    # Simple concatenation: wrap value in a list and cat
    try:
        return lst.cat(ee.List([value]))
    except Exception as e:
        lst_type = type(lst).__name__
        value_type = type(value).__name__
        logger.error(
            "cat_one: FAILED at line=ee_utils.py:cat_one | "
            "list type=%s, value type=%s, error=%s: %s",
            lst_type,
            value_type,
            type(e).__name__,
            str(e),
            exc_info=True
        )
        raise


def remove_nulls(lst: Any) -> ee.List:
    """Remove null values from a list using client-side type checking.
    
    Uses Python list filtering when the input is a Python list, otherwise
    delegates to ee.List filtering. This provides better performance for
    Python lists while maintaining compatibility with ee.List objects.
    
    Args:
        lst: An ee.List or Python list potentially containing null values
        
    Returns:
        An ee.List with null values removed
        
    Examples:
        >>> remove_nulls([1, None, 2, None, 3])  # Returns ee.List([1, 2, 3])
        >>> remove_nulls(ee.List([1, None, 2]))  # Returns ee.List([1, 2])
    """
    # Client-side check: if it's a Python list, filter client-side then convert
    if isinstance(lst, (list, tuple)):
        filtered = [x for x in lst if x is not None]
        return ee.List(filtered)
    
    # Ensure it's an ee.List and use server-side filtering
    is_ee_list = False
    try:
        is_ee_list = isinstance(lst, ee.List)
    except TypeError:
        # ee.List is not a proper type (might be mocked), check by class name
        is_ee_list = lst.__class__.__name__ == 'List' and hasattr(lst, 'filter')
    
    if not is_ee_list:
        try:
            lst = ee.List(lst)
        except (TypeError, AttributeError) as e:
            lst_type = type(lst).__name__
            logger.warning(
                "remove_nulls: ee.List(lst) failed with %s: %s at type=%s, trying ensure_list",
                type(e).__name__,
                str(e),
                lst_type
            )
            # Try using ensure_list as fallback
            try:
                lst = ensure_list(lst)
            except Exception as fallback_error:
                logger.error(
                    "remove_nulls: ALL FALLBACKS FAILED at line=ee_utils.py:remove_nulls | "
                    "input type=%s, error=%s: %s",
                    lst_type,
                    type(fallback_error).__name__,
                    str(fallback_error),
                    exc_info=True
                )
                raise
    
    try:
        return lst.filter(ee.Filter.notNull(['item']))
    except Exception as e:
        logger.error(
            "remove_nulls: filter failed at line=ee_utils.py:remove_nulls | "
            "list type=%s, error=%s: %s",
            type(lst).__name__,
            type(e).__name__,
            str(e),
            exc_info=True
        )
        raise
