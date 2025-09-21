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

from app.services import tiles  # noqa: E402  pylint: disable=wrong-import-position
from app.services.indices import resolve_index  # noqa: E402  pylint: disable=wrong-import-position


def test_gndvi_images_and_metadata(monkeypatch):
    context = setup_fake_ee(monkeypatch, tiles, [0.4, 0.8])

    definition, params = resolve_index("gndvi")

    annual = tiles.index_annual_image(
        {"type": "Polygon", "coordinates": []},
        2020,
        definition=definition,
        parameters=params,
    )

    assert isinstance(annual, FakeMeanImage)
    assert annual.clamped_to == (-1.0, 1.0)
    assert annual.value == pytest.approx(0.6)

    bands = context["log"].get("normalized_difference_bands", [])
    assert bands, "Expected normalized difference bands to be recorded"
    assert set(bands) == {("B8", "B3")}

    context["values"] = [1.5, 2.0]
    month = tiles.index_month_image(
        {"type": "Polygon", "coordinates": []},
        2020,
        6,
        definition=definition,
        parameters=params,
    )

    assert isinstance(month, FakeMeanImage)
    assert month.clamped_to == (-1.0, 1.0)
    assert month.value == pytest.approx(1.0)

    tile = tiles.get_tile_template_for_image(month, definition=definition)
    assert tile["vis"]["bands"] == [definition.band_name]
    assert tile["vis"]["min"] == pytest.approx(definition.valid_range[0])
    assert tile["vis"]["max"] == pytest.approx(definition.valid_range[1])
    assert tile["vis"]["palette"] == list(definition.default_palette)


def test_index_tile_visualization_overrides(monkeypatch):
    setup_fake_ee(monkeypatch, tiles, [0.1, 0.5])
    definition, params = resolve_index("gndvi")

    image = tiles.index_month_image(
        {"type": "Polygon", "coordinates": []},
        2021,
        4,
        definition=definition,
        parameters=params,
    )

    overrides = {"min": 0.0, "max": 0.8, "palette": "00FF00,FFFFFF"}
    clamp_range = tiles.resolve_clamp_range(definition, overrides)
    assert clamp_range == (0.0, 0.8)

    tile = tiles.get_tile_template_for_image(
        image,
        definition=definition,
        vis_overrides=overrides,
    )

    assert tile["vis"]["min"] == pytest.approx(0.0)
    assert tile["vis"]["max"] == pytest.approx(0.8)
    assert tile["vis"]["palette"] == ["00FF00", "FFFFFF"]

