from __future__ import annotations

import ee

def cat_one(lst, value):
    # Safely concatenate by wrapping value in a list
    return ee.List(lst).cat(ee.List([value]))

def ensure_list(value):
    # Safely ensure value is a list, wrapping scalars, and RETURN an ee.List
    obj_type = ee.String(ee.Algorithms.ObjectType(value))
    is_list = ee.String(obj_type).compareTo(ee.String('List')).eq(0)
    return ee.List(ee.Algorithms.If(is_list, value, ee.List([value])))


def ensure_number(value):
    obj_type = ee.String(ee.Algorithms.ObjectType(value))
    is_num = ee.String(obj_type).compareTo(ee.String('Number')).eq(0)
    return ee.Algorithms.If(is_num, value, ee.Number(value))
    
def remove_nulls(lst):
    # Filter out nulls from the list
    return ee.List(lst).filter(ee.Filter.notNull(['item']))
