"""Compatibility shim that prefers the real SciPy package when available."""
from __future__ import annotations

from types import ModuleType
import importlib.machinery
import importlib.util
import site
import sys

__all__: list[str] = []


def _load_real_scipy() -> ModuleType | None:
    search_paths: list[str] = []
    try:
        search_paths.extend(site.getsitepackages())
    except Exception:  # pragma: no cover - platform differences
        pass
    try:
        user_site = site.getusersitepackages()
    except Exception:  # pragma: no cover - platform differences
        user_site = None
    if user_site:
        search_paths.append(user_site)

    for base in search_paths:
        spec = importlib.machinery.PathFinder.find_spec(__name__, [base])
        if spec is None:
            continue
        origin = getattr(spec, "origin", "")
        if origin and origin == __file__:
            continue
        loader = spec.loader
        if loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        previous = sys.modules.get(__name__)
        sys.modules[__name__] = module
        try:
            loader.exec_module(module)
        except Exception:
            if previous is not None:
                sys.modules[__name__] = previous
            else:
                sys.modules.pop(__name__, None)
            continue
        return module
    return None


_REAL_SCIPY = _load_real_scipy()

if _REAL_SCIPY is not None:
    sys.modules[__name__] = _REAL_SCIPY
    for attr in dir(_REAL_SCIPY):
        if attr.startswith("__"):
            continue
        if attr == "ndimage":
            continue
        globals()[attr] = getattr(_REAL_SCIPY, attr)
    try:
        from . import ndimage as _ndimage_stub
    except Exception:  # pragma: no cover - fallback when stub missing
        globals()["ndimage"] = getattr(_REAL_SCIPY, "ndimage")
    else:
        globals()["ndimage"] = _ndimage_stub
    __all__ = list(getattr(_REAL_SCIPY, "__all__", []))
    if "ndimage" not in __all__:
        __all__.append("ndimage")
else:  # pragma: no cover - exercised implicitly in CI without SciPy
    from . import ndimage  # type: ignore  # noqa: F401
    __all__ = ["ndimage"]

del _REAL_SCIPY
del _load_real_scipy
