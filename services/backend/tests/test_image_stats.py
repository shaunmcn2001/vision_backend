import math
from pathlib import Path
import sys

import pytest

TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    sys.path.append(str(TEST_DIR))

from fake_ee import FakeMeanImage, setup_fake_ee  # noqa: E402  pylint: disable=wrong-import-position

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

from app.services import image_stats  # noqa: E402  pylint: disable=wrong-import-position


def test_temporal_stats_uses_sum_and_count(monkeypatch):
    context = setup_fake_ee(monkeypatch, image_stats, [0.2, 0.4, 0.6])

    collection = image_stats.ee.ImageCollection("TEST_COLLECTION")
    stats = image_stats.temporal_stats(
        collection,
        band_name="NDVI",
        rename_prefix="NDVI",
        mean_band_name="NDVI",
    )

    assert isinstance(stats["mean"], FakeMeanImage)
    assert stats["mean"].name == "NDVI"
    assert stats["mean"].value == pytest.approx(0.4)
    assert stats["median"].name == "NDVI_median"
    assert stats["median"].value == pytest.approx(0.4)
    assert stats["std"].name == "NDVI_stdDev"
    expected_std = math.sqrt(((0.2 - 0.4) ** 2 + (0.4 - 0.4) ** 2 + (0.6 - 0.4) ** 2) / 3)
    assert stats["std"].value == pytest.approx(expected_std)
    assert stats["cv"].name == "NDVI_cv"
    assert stats["cv"].value == pytest.approx(expected_std / 0.4)

    assert stats["raw_sum"].value == pytest.approx(1.2)
    assert stats["raw_count"].value == 3
    assert stats["valid_mask"].value == 1
    assert isinstance(stats["mean"].mask, FakeMeanImage)
    assert stats["mean"].mask.value == 1

    reduce_calls = context["log"].get("reduce_calls", [])
    assert reduce_calls == ["sum", "count", "median", "stdDev"]
