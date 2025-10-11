# services/backend/app/services/ee_utils.py
from __future__ import annotations
import ee

def _ensure_list(value):
    obj_type = ee.String(ee.Algorithms.ObjectType(value))  # Force ee.String to avoid client str
    is_list = obj_type.eq('List')
    return ee.Algorithms.If(is_list, value, ee.List([value]))

def _cat_one(lst, value):
    return ee.List(lst).cat(ee.List([value]))

def ensure_number(x):
    """Normalize expression/scalar to ee.Number."""
    return ee.Number(x)

def ensure_string(x):
    """Normalize to ee.String."""
    return ee.String(x)

def remove_nulls(lst):
    """Remove nulls from an ee.List (Filter.notNull is for Collections)."""
    return ee.List(lst).removeAll([None])
