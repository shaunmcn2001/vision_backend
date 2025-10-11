from __future__ import annotations

import ee

def cat_one(lst, value):
    # Safely concatenate by wrapping value in a list
    return ee.List(lst).cat(ee.List([value]))

def ensure_list(value):
    # Safely ensure value is a list, wrapping scalars
    obj_type = ee.String(ee.Algorithms.ObjectType(value))
    is_list = obj_type.eq('List')
    return ee.Algorithms.If(is_list, value, ee.List([value]))

def ensure_number(value):
    # Convert to ee.Number
    return ee.Number(value)

def remove_nulls(lst):
    # Filter out nulls from the list
    return ee.List(lst).filter(ee.Filter.notNull(['item']))
