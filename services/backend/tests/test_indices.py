import math
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

from app import indices

SAMPLE_BANDS = {
    "B2": 0.12,
    "B3": 0.18,
    "B4": 0.20,
    "B5": 0.26,
    "B6": 0.27,
    "B7": 0.28,
    "B8": 0.60,
    "B8A": 0.62,
    "B11": 0.32,
    "B12": 0.29,
}


msavi_expected = (
    (2 * SAMPLE_BANDS["B8"] + 1)
    - math.sqrt(
        (2 * SAMPLE_BANDS["B8"] + 1) ** 2
        - 8 * (SAMPLE_BANDS["B8"] - SAMPLE_BANDS["B4"])
    )
) / 2


@pytest.mark.parametrize(
    "index_name,expected",
    [
        (
            "NDVI",
            (SAMPLE_BANDS["B8"] - SAMPLE_BANDS["B4"])
            / (SAMPLE_BANDS["B8"] + SAMPLE_BANDS["B4"]),
        ),
        (
            "EVI",
            2.5
            * (SAMPLE_BANDS["B8"] - SAMPLE_BANDS["B4"])
            / (
                SAMPLE_BANDS["B8"]
                + 6 * SAMPLE_BANDS["B4"]
                - 7.5 * SAMPLE_BANDS["B2"]
                + 1
            ),
        ),
        ("MSAVI", msavi_expected),
        (
            "NDMI",
            (SAMPLE_BANDS["B8"] - SAMPLE_BANDS["B11"])
            / (SAMPLE_BANDS["B8"] + SAMPLE_BANDS["B11"]),
        ),
        (
            "BSI",
            (
                SAMPLE_BANDS["B11"]
                + SAMPLE_BANDS["B4"]
                - (SAMPLE_BANDS["B8"] + SAMPLE_BANDS["B2"])
            )
            / (
                SAMPLE_BANDS["B11"]
                + SAMPLE_BANDS["B4"]
                + (SAMPLE_BANDS["B8"] + SAMPLE_BANDS["B2"])
            ),
        ),
    ],
)
def test_scalar_indices_match_manual(index_name, expected):
    result = indices.compute_scalar_index(index_name, SAMPLE_BANDS)
    if math.isnan(expected):
        assert math.isnan(result)
    else:
        assert result == pytest.approx(expected, rel=1e-6)


def test_scalar_index_nan_on_zero_division():
    values = SAMPLE_BANDS.copy()
    values["B8"] = 0.2
    values["B4"] = -0.2  # denominator becomes zero
    result = indices.compute_scalar_index("NDVI", values)
    assert math.isnan(result)


def test_supported_indices_matches_expected():
    expected = {
        "NDVI",
        "EVI",
        "GNDVI",
        "NDRE",
        "SAVI",
        "MSAVI",
        "VARI",
        "MCARI",
        "NDWI_McFeeters",
        "NDWI_Gao",
        "NDMI",
        "MSI",
        "GVMI",
        "NBR",
        "PSRI",
        "ARI",
        "CRI",
        "BSI",
        "SBI",
        "NDSI_Soil",
        "NDTI",
        "PRI",
    }
    assert set(indices.SUPPORTED_INDICES) == expected


def test_compute_scalar_index_unsupported():
    with pytest.raises(KeyError):
        indices.compute_scalar_index("NOT_REAL", SAMPLE_BANDS)
