import sys
from pathlib import Path

import pytest

ee = pytest.importorskip("ee")

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

from app.services.zones_core import ensure_list, remove_nulls, as_number

try:  # pragma: no cover - skip if EE can't initialise in CI
    ee.Initialize()
except Exception:  # pragma: no cover - environment guard
    pytest.skip("Earth Engine initialization failed", allow_module_level=True)


def test_ensure_list_scalar_and_list():
    assert ee.List(ensure_list(ee.Number(5))).size().getInfo() == 1
    assert ee.List(ensure_list(ee.List([1, 2, 3]))).size().getInfo() == 3


def test_remove_nulls():
    y = remove_nulls(ee.List([1, None, 2, None, 3]))
    assert ee.List(y).size().getInfo() == 3


def test_as_number_if():
    n = as_number(ee.Algorithms.If(ee.Number(1).eq(1), 1, 0))
    assert ee.Number(n).getInfo() in (0, 1)
