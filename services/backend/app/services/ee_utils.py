from __future__ import annotations

import ee


def cat_one(lst, value):
    # Safely concatenate by wrapping value in a list
    return ee.List(lst).cat(ee.List([value]))


def ensure_list(value):
    """Safely ensure *value* behaves like an ee.List."""

    # Handle native Python iterables before touching Earth Engine APIs.
    if isinstance(value, (list, tuple)):
        return ee.List(list(value))

    # If value is an EE computed object, inspect its type and wrap as needed.
    if isinstance(value, ee.computedobject.ComputedObject):
        obj_type = ee.String(ee.Algorithms.ObjectType(value))
        is_list = obj_type.eq(ee.String("List"))
        return ee.Algorithms.If(is_list, value, ee.List([value]))

    # Fallback: treat anything else as a scalar and wrap it.
    return ee.List([value])

def ensure_number(value):
    # Convert to ee.Number
    return ee.Number(value)

def remove_nulls(lst):
    # Filter out nulls from the list
    return ee.List(lst).filter(ee.Filter.notNull(['item']))
