# services/backend/app/services/ee_utils.py
from __future__ import annotations
import ee

def ensure_list(x):
    """
    EE-safe: return an ee.List whether x is scalar or already a list.
    Implementation: wrap then flatten.
      - If x is ee.List -> [x].flatten() == x
      - If x is scalar (ee.Number/String/bool/int/float) -> [x]
    """
    return ee.List([x]).flatten()

def ensure_number(x):
    """Normalize expression/scalar to ee.Number."""
    return ee.Number(x)

def ensure_string(x):
    """Normalize to ee.String."""
    return ee.String(x)

def remove_nulls(lst):
    """Remove nulls from an ee.List (Filter.notNull is for Collections)."""
    return ee.List(lst).removeAll([None])

def cat_one(lst, value):
    """Append a single value (scalar or list) to an ee.List safely."""
    return ee.List(lst).cat(ee.List([value]))
