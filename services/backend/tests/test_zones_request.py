from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    sys.path.append(str(TEST_DIR))

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

from app.api.zones import ProductionZonesRequest


def _base_payload() -> dict:
    return {
        "aoi_geojson": {"type": "Polygon", "coordinates": []},
        "aoi_name": "example",
    }


def test_production_zones_request_accepts_month_list():
    payload = {
        **_base_payload(),
        "months": ["2021-01", "2021-02", "2021-03"],
    }

    request = ProductionZonesRequest(**payload)

    assert request.months == ["2021-01", "2021-02", "2021-03"]
    assert request.start_month is None
    assert request.end_month is None


def test_production_zones_request_accepts_start_end_range():
    payload = {
        **_base_payload(),
        "start_month": "2021-01",
        "end_month": "2021-06",
    }

    request = ProductionZonesRequest(**payload)

    assert request.months == []
    assert request.start_month == "2021-01"
    assert request.end_month == "2021-06"


def test_production_zones_request_requires_months_or_range():
    payload = _base_payload()

    with pytest.raises(ValidationError) as excinfo:
        ProductionZonesRequest(**payload)

    assert "Supply either months[] or start_month/end_month" in str(excinfo.value)
