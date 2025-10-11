import ee


def ensure_list(x):
    """
    EE-safe: returns an ee.List whether x is scalar or already a list.
    - If x is ee.List: flatten returns the list.
    - If x is scalar (ee.Number/String/bool/int/float), wraps in a single-element list.
    """
    try:
        return ee.List(x)
    except Exception:  # pragma: no cover - relies on EE server behaviour
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


def ensure_number(x):
    """
    Normalize any scalar-like expression to ee.Number.
    Useful when condensing boolean/if results into numbers.
    """
    return ee.Number(x)


def remove_nulls(lst):
    """
    Remove nulls from an ee.List.
    .filter(ee.Filter.notNull(...)) is for collections, not lists.
    """
    return ee.List(lst).removeAll([None])


def cat_one(lst, value):
    """
    Append a single value (scalar or list) to an ee.List safely.
    Converts scalars to a one-element list via ensure_list() before concatenation.
    """
    return ee.List(lst).cat(ensure_list(value))
