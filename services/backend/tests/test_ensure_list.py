import sys
from pathlib import Path

import pytest

ee = pytest.importorskip("ee")

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

import app.services.ee_utils as ee_utils  # noqa: E402

ensure_list = ee_utils.ensure_list
ensure_number = ee_utils.ensure_number
remove_nulls = ee_utils.remove_nulls
cat_one = ee_utils.cat_one

try:  # pragma: no cover - skip if EE can't initialise in CI
    ee.Initialize()
except Exception:  # pragma: no cover - environment guard
    pytest.skip("Earth Engine initialization failed", allow_module_level=True)


def test_ensure_list_scalar_and_list():
    assert ee.List(ensure_list(ee.Number(5))).size().getInfo() == 1
    assert ee.List(ensure_list(ee.List([1, 2, 3]))).size().getInfo() == 3


def test_ensure_list_handles_if_scalar():
    conditional = ee.Algorithms.If(ee.Number(1).eq(1), 1, 0)
    lst = ensure_list(conditional)
    assert ee.List(lst).size().getInfo() == 1


def test_remove_nulls():
    y = remove_nulls(ee.List([1, None, 2, None, 3]))
    assert ee.List(y).size().getInfo() == 3


def test_cat_one_with_scalar():
    lst = cat_one([], 1)
    assert ee.List(lst).size().getInfo() == 1
    assert ee.List(lst).get(0).getInfo() == 1


def test_ensure_number_on_if():
    n = ensure_number(ee.Algorithms.If(ee.Number(1).eq(1), 1, 0))
    assert ee.Number(n).getInfo() in (0, 1)
