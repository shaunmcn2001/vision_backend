"""Pytest fixtures and fakes."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
FAKE_RASTERIO_PATH = TESTS_DIR / "fakes" / "fake_rasterio.py"

if "tests.fake_rasterio" in sys.modules:
    fake_rasterio = sys.modules["tests.fake_rasterio"]
else:
    spec = importlib.util.spec_from_file_location(
        "tests.fake_rasterio", FAKE_RASTERIO_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load fake_rasterio module")
    fake_rasterio = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fake_rasterio
    spec.loader.exec_module(fake_rasterio)

sys.modules.setdefault("rasterio", fake_rasterio)
sys.modules.setdefault("rasterio.transform", fake_rasterio.transform)
sys.modules.setdefault("rasterio.features", fake_rasterio.features)
