"""Ensure the backend service package is importable from the repository root.

Render and other PaaS providers often run the start command from the repo root,
which means the `services/backend` directory containing the FastAPI package is
not automatically on `sys.path`. By appending it here we allow `uvicorn
app.main:app` (and similar entrypoints) to resolve correctly without requiring
manual PYTHONPATH tweaks while keeping system-installed packages ahead of the
repository modules.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent / "services" / "backend"

if _BACKEND_DIR.is_dir():
    backend_path = str(_BACKEND_DIR)
    if backend_path not in sys.path:
        sys.path.append(backend_path)
