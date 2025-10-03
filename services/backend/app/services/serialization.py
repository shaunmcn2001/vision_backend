"""Utilities for sanitising Earth Engine metadata payloads."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Iterable

try:  # pragma: no cover - fallback when the library is unavailable
    import ee  # type: ignore
except Exception:  # pragma: no cover - allow running without ee installed
    ee = None  # type: ignore[assignment]


_MODULE_TYPE_HINTS = {
    "image": "ee.Image",
    "imagecollection": "ee.ImageCollection",
    "computedobject": "ee.ComputedObject",
    "feature": "ee.Feature",
    "featurecollection": "ee.FeatureCollection",
    "geometry": "ee.Geometry",
    "dictionary": "ee.Dictionary",
    "list": "ee.List",
    "number": "ee.Number",
    "string": "ee.String",
}


def _module_type_name(obj: Any) -> str:
    module = getattr(obj.__class__, "__module__", "")
    module_tail = module.split(".")[-1] if module else ""
    return _MODULE_TYPE_HINTS.get(module_tail.lower(), f"ee.{obj.__class__.__name__}")


def _is_ee_object(value: Any) -> bool:
    if value is None:
        return False
    module = getattr(value.__class__, "__module__", "")
    if module.startswith("ee.") or module == "ee":
        return True
    computed = getattr(ee, "ComputedObject", None) if ee is not None else None
    return isinstance(value, computed) if isinstance(computed, type) else False


def _extract_band_names(image: Any) -> list[str] | None:
    bands: Iterable[Any] | None = None
    try:
        band_obj = image.bandNames()
    except Exception:  # pragma: no cover - defensive guard
        return None

    if hasattr(band_obj, "getInfo"):
        try:
            info = band_obj.getInfo()
        except Exception:  # pragma: no cover - guard against uninitialised EE
            info = None
    else:
        info = band_obj

    if isinstance(info, Mapping):  # pragma: no cover - unexpected structure
        return None

    if isinstance(info, (list, tuple, set)):
        bands = list(info)
    elif isinstance(info, Iterable) and not isinstance(info, (str, bytes)):
        bands = list(info)
    else:
        bands = None

    if bands is None:
        return None

    return [str(name) for name in bands]


def _describe_ee_object(value: Any) -> Any:
    """Return a primitive-friendly representation of an EE object."""
    type_name = _module_type_name(value)
    if type_name == "ee.Image":
        bands = _extract_band_names(value)
        if bands is not None:
            return {"type": type_name, "bands": bands}
        return {"type": type_name}
    return {"type": type_name}


def sanitize_metadata(payload: Any) -> Any:
    """Recursively sanitise metadata payloads for JSON serialisation."""
    if _is_ee_object(payload):
        return _describe_ee_object(payload)
    if isinstance(payload, Mapping):
        return {key: sanitize_metadata(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple, set)):
        return [sanitize_metadata(item) for item in payload]
    return payload


__all__ = ["sanitize_metadata"]
