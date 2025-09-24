"""Proxy package so `import app` resolves to the backend service package.

The actual FastAPI application lives under ``services/backend/app``. Platforms
like Render import ``app.main`` from the repository root, which previously
failed with ``ModuleNotFoundError``. By customising ``__path__`` we delegate all
submodule lookups to the real package without duplicating code or adjusting
start commands.
"""
from __future__ import annotations

from pathlib import Path

_backend_package = (
    Path(__file__).resolve().parent.parent / "services" / "backend" / "app"
)

if not _backend_package.is_dir():
    raise ImportError(
        "The backend application package was not found at"
        f" {_backend_package!s}."
    )

__path__ = [str(_backend_package)]
__file__ = str(_backend_package / "__init__.py")
