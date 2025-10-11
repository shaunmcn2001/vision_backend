"""Helpers for normalising nested payloads before JSON serialisation."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

try:  # pragma: no cover - optional dependency for unit tests
    import ee  # type: ignore
except Exception:  # pragma: no cover - allow running without ee installed
    ee = None  # type: ignore[assignment]


def _is_ee_object(value: Any) -> bool:
    if value is None:
        return False
    module = getattr(value.__class__, "__module__", "")
    if module.startswith("ee.") or module == "ee":
        return True
    computed = getattr(ee, "ComputedObject", None) if ee is not None else None
    return isinstance(value, computed) if isinstance(computed, type) else False


def sanitize_for_json(value: Any) -> Any:
    """Recursively sanitise a payload so that it can be JSON serialised."""

    if _is_ee_object(value):
        return None

    get_info = getattr(value, "getInfo", None)
    if callable(get_info):
        try:
            info = get_info()
        except Exception:  # pragma: no cover - guard against failed lookups
            info = None
        return sanitize_for_json(info)

    if is_dataclass(value) and not isinstance(value, type):
        return sanitize_for_json(asdict(value))

    if isinstance(value, Mapping):
        return {key: sanitize_for_json(item) for key, item in value.items()}

    if hasattr(value, "__dict__") and not isinstance(value, type):
        return {key: sanitize_for_json(item) for key, item in vars(value).items()}

    if isinstance(value, (tuple, set)):
        return [sanitize_for_json(item) for item in value]

    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        return [sanitize_for_json(item) for item in value]

    return value


__all__ = ["sanitize_for_json"]

