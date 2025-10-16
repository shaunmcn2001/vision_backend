from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

backend_dir = Path(__file__).resolve().parents[1]
backend_dir_str = str(backend_dir)
if backend_dir_str not in sys.path:
    sys.path.append(backend_dir_str)

from app.services.zones import _stream_zonal_stats


def test_stream_zonal_stats_ignores_nodata_and_masked_pixels(tmp_path):
    transform = from_origin(0, 0, 1, 1)

    zones_path = tmp_path / "zones.tif"
    zones_data = np.array(
        [[1, 1, 1, 1], [1, 1, 2, 2]],
        dtype=np.uint8,
    )
    with rasterio.open(
        zones_path,
        "w",
        driver="GTiff",
        height=zones_data.shape[0],
        width=zones_data.shape[1],
        count=1,
        dtype="uint8",
        transform=transform,
    ) as dst:
        dst.write(zones_data, 1)

    ndvi_path = tmp_path / "ndvi.tif"
    ndvi_data = np.array(
        [[0.1, 0.2, -9999.0, np.inf], [0.3, -9999.0, 0.5, 0.6]],
        dtype=np.float32,
    )
    ndvi_mask = np.full(ndvi_data.shape, 255, dtype=np.uint8)
    ndvi_mask[0, 2] = 0  # simulate GDAL mask flagging nodata pixels
    with rasterio.open(
        ndvi_path,
        "w",
        driver="GTiff",
        height=ndvi_data.shape[0],
        width=ndvi_data.shape[1],
        count=1,
        dtype="float32",
        nodata=-9999.0,
        transform=transform,
    ) as dst:
        dst.write(ndvi_data, 1)
        dst.write_mask(ndvi_mask)

    stats = _stream_zonal_stats(zones_path, ndvi_path)

    assert [entry["zone"] for entry in stats] == [1, 2]

    zone1 = stats[0]
    assert zone1["pixel_count"] == 3
    assert zone1["mean_ndvi"] == pytest.approx((0.1 + 0.2 + 0.3) / 3)
    assert zone1["min_ndvi"] == pytest.approx(0.1)
    assert zone1["max_ndvi"] == pytest.approx(0.3)
    assert zone1["area_ha"] == pytest.approx(3 / 10_000)

    zone2 = stats[1]
    assert zone2["pixel_count"] == 2
    assert zone2["mean_ndvi"] == pytest.approx((0.5 + 0.6) / 2)
    assert zone2["min_ndvi"] == pytest.approx(0.5)
    assert zone2["max_ndvi"] == pytest.approx(0.6)
    assert zone2["area_ha"] == pytest.approx(2 / 10_000)
