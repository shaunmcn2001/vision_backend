import sys
from pathlib import Path

import pytest

TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    sys.path.append(str(TEST_DIR))

from fake_ee import FakeMeanImage, setup_fake_ee


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

from app.api import export
from app.services import tiles


def test_export_ndvi_range_preserves_negatives(monkeypatch):
    context = setup_fake_ee(monkeypatch, export, [-0.6, -0.2])

    definition, params = export.resolve_index("ndvi")
    _, image = export._index_image_for_range(
        {"type": "Point", "coordinates": [0, 0]},
        "2024-01-01",
        "2024-02-01",
        definition=definition,
        parameters=params,
    )

    assert isinstance(image, FakeMeanImage)
    assert image.clamped_to == (-1, 1)
    assert image.value == pytest.approx(-0.4)

    # Updating context should not affect the already computed image
    context["values"] = [-0.1, 0.2]
    assert image.value == pytest.approx(-0.4)


def test_tile_ndvi_images_allow_negative_values(monkeypatch):
    context = setup_fake_ee(monkeypatch, tiles, [-0.8, -0.4])

    annual = tiles.ndvi_annual_image({"type": "Polygon", "coordinates": []}, 2021)
    assert isinstance(annual, FakeMeanImage)
    assert annual.clamped_to == (-1, 1)
    assert annual.value == pytest.approx(-0.6)

    context["values"] = [-0.7, 0.1]
    month = tiles.ndvi_month_image({"type": "Polygon", "coordinates": []}, 2021, 5)
    assert isinstance(month, FakeMeanImage)
    assert month.clamped_to == (-1, 1)
    assert month.value == pytest.approx(-0.3)
