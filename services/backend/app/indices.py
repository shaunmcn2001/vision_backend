"""Utilities for computing Sentinel-2 indices using Earth Engine images.

This module provides helpers to build index images at a consistent scale and
sanity-check utilities that can be evaluated locally for unit tests without an
Earth Engine session.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Mapping
import math

import ee

TEN_METER_BANDS = {"B2", "B3", "B4", "B8"}
TWENTY_METER_BANDS = {"B5", "B6", "B7", "B8A", "B11", "B12"}
REQUIRED_BANDS: Iterable[str] = (
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B11",
    "B12",
)


def _select_band(image: ee.Image, band: str) -> ee.Image:
    """Select a band and promote 20 m bands to 10 m using bilinear resampling."""
    selected = image.select(band)
    if band in TWENTY_METER_BANDS:
        selected = selected.resample("bilinear")
    return selected


def _safe_divide(numerator: ee.Image, denominator: ee.Image) -> ee.Image:
    mask = denominator.abs().gt(0)
    numerator_masked = numerator.updateMask(mask)
    denominator_masked = denominator.updateMask(mask)
    return numerator_masked.divide(denominator_masked)


def _safe_sqrt(image: ee.Image) -> ee.Image:
    mask = image.gte(0)
    return image.updateMask(mask).sqrt()


def _finish(
    image: ee.Image, name: str, geometry: ee.Geometry, scale_m: int
) -> ee.Image:
    wrapped = ee.Image(image)
    return (
        wrapped.rename(name)
        .toFloat()
        .clip(geometry)
        .reproject("EPSG:4326", None, scale_m)
    )


IndexFunction = Callable[[ee.Image, ee.Geometry, int], ee.Image]


def ndvi(image: ee.Image, geometry: ee.Geometry, scale_m: int) -> ee.Image:
    nir = _select_band(image, "B8")
    red = _select_band(image, "B4")
    result = _safe_divide(nir.subtract(red), nir.add(red))
    return _finish(result, "NDVI", geometry, scale_m)


def evi(image: ee.Image, geometry: ee.Geometry, scale_m: int) -> ee.Image:
    nir = _select_band(image, "B8")
    red = _select_band(image, "B4")
    blue = _select_band(image, "B2")
    denominator = nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1)
    result = _safe_divide(nir.subtract(red).multiply(2.5), denominator)
    return _finish(result, "EVI", geometry, scale_m)


def gndvi(image: ee.Image, geometry: ee.Geometry, scale_m: int) -> ee.Image:
    nir = _select_band(image, "B8")
    green = _select_band(image, "B3")
    result = _safe_divide(nir.subtract(green), nir.add(green))
    return _finish(result, "GNDVI", geometry, scale_m)


def ndre(image: ee.Image, geometry: ee.Geometry, scale_m: int) -> ee.Image:
    nir = _select_band(image, "B8")
    re1 = _select_band(image, "B5")
    result = _safe_divide(nir.subtract(re1), nir.add(re1))
    return _finish(result, "NDRE", geometry, scale_m)


def savi(image: ee.Image, geometry: ee.Geometry, scale_m: int) -> ee.Image:
    nir = _select_band(image, "B8")
    red = _select_band(image, "B4")
    numerator = nir.subtract(red)
    denominator = nir.add(red).add(0.5)
    result = _safe_divide(numerator, denominator).multiply(1.5)
    return _finish(result, "SAVI", geometry, scale_m)


def msavi(image: ee.Image, geometry: ee.Geometry, scale_m: int) -> ee.Image:
    nir = _select_band(image, "B8")
    red = _select_band(image, "B4")
    two_nir_plus_one = nir.multiply(2).add(1)
    radicand = two_nir_plus_one.pow(2).subtract(nir.subtract(red).multiply(8))
    sqrt_term = _safe_sqrt(radicand)
    result = two_nir_plus_one.subtract(sqrt_term).divide(2)
    return _finish(result, "MSAVI", geometry, scale_m)


def vari(image: ee.Image, geometry: ee.Geometry, scale_m: int) -> ee.Image:
    green = _select_band(image, "B3")
    red = _select_band(image, "B4")
    blue = _select_band(image, "B2")
    numerator = green.subtract(red)
    denominator = green.add(red).subtract(blue)
    result = _safe_divide(numerator, denominator)
    return _finish(result, "VARI", geometry, scale_m)


def mcari(image: ee.Image, geometry: ee.Geometry, scale_m: int) -> ee.Image:
    re1 = _select_band(image, "B5")
    red = _select_band(image, "B4")
    green = _select_band(image, "B3")
    term1 = re1.subtract(red)
    term2 = re1.subtract(green).multiply(0.2)
    ratio = _safe_divide(re1, red)
    result = term1.subtract(term2).multiply(ratio)
    return _finish(result, "MCARI", geometry, scale_m)


def ndwi_mcfeeters(image: ee.Image, geometry: ee.Geometry, scale_m: int) -> ee.Image:
    green = _select_band(image, "B3")
    nir = _select_band(image, "B8")
    result = _safe_divide(green.subtract(nir), green.add(nir))
    return _finish(result, "NDWI_McFeeters", geometry, scale_m)


def ndwi_gao(image: ee.Image, geometry: ee.Geometry, scale_m: int) -> ee.Image:
    nir = _select_band(image, "B8")
    swir1 = _select_band(image, "B11")
    result = _safe_divide(nir.subtract(swir1), nir.add(swir1))
    return _finish(result, "NDWI_Gao", geometry, scale_m)


def ndmi(image: ee.Image, geometry: ee.Geometry, scale_m: int) -> ee.Image:
    nir = _select_band(image, "B8")
    swir1 = _select_band(image, "B11")
    result = _safe_divide(nir.subtract(swir1), nir.add(swir1))
    return _finish(result, "NDMI", geometry, scale_m)


def msi(image: ee.Image, geometry: ee.Geometry, scale_m: int) -> ee.Image:
    swir1 = _select_band(image, "B11")
    nir = _select_band(image, "B8")
    result = _safe_divide(swir1, nir)
    return _finish(result, "MSI", geometry, scale_m)


def gvmi(image: ee.Image, geometry: ee.Geometry, scale_m: int) -> ee.Image:
    nir = _select_band(image, "B8")
    swir1 = _select_band(image, "B11")
    numerator = nir.add(0.1).subtract(swir1.add(0.02))
    denominator = nir.add(0.1).add(swir1.add(0.02))
    result = _safe_divide(numerator, denominator)
    return _finish(result, "GVMI", geometry, scale_m)


def nbr(image: ee.Image, geometry: ee.Geometry, scale_m: int) -> ee.Image:
    nir = _select_band(image, "B8")
    swir2 = _select_band(image, "B12")
    result = _safe_divide(nir.subtract(swir2), nir.add(swir2))
    return _finish(result, "NBR", geometry, scale_m)


def psri(image: ee.Image, geometry: ee.Geometry, scale_m: int) -> ee.Image:
    red = _select_band(image, "B4")
    green = _select_band(image, "B3")
    re2 = _select_band(image, "B6")
    result = _safe_divide(red.subtract(green), re2)
    return _finish(result, "PSRI", geometry, scale_m)


def ari(image: ee.Image, geometry: ee.Geometry, scale_m: int) -> ee.Image:
    green = _select_band(image, "B3")
    re1 = _select_band(image, "B5")
    one = ee.Image.constant(1)
    result = _safe_divide(one, green).subtract(_safe_divide(one, re1))
    return _finish(result, "ARI", geometry, scale_m)


def cri(image: ee.Image, geometry: ee.Geometry, scale_m: int) -> ee.Image:
    blue = _select_band(image, "B2")
    re1 = _select_band(image, "B5")
    one = ee.Image.constant(1)
    result = _safe_divide(one, blue).subtract(_safe_divide(one, re1))
    return _finish(result, "CRI", geometry, scale_m)


def bsi(image: ee.Image, geometry: ee.Geometry, scale_m: int) -> ee.Image:
    swir1 = _select_band(image, "B11")
    red = _select_band(image, "B4")
    nir = _select_band(image, "B8")
    blue = _select_band(image, "B2")
    numerator = swir1.add(red).subtract(nir.add(blue))
    denominator = swir1.add(red).add(nir.add(blue))
    result = _safe_divide(numerator, denominator)
    return _finish(result, "BSI", geometry, scale_m)


def sbi(image: ee.Image, geometry: ee.Geometry, scale_m: int) -> ee.Image:
    red = _select_band(image, "B4")
    green = _select_band(image, "B3")
    radicand = red.pow(2).add(green.pow(2)).divide(2)
    result = _safe_sqrt(radicand)
    return _finish(result, "SBI", geometry, scale_m)


def ndsi_soil(image: ee.Image, geometry: ee.Geometry, scale_m: int) -> ee.Image:
    swir1 = _select_band(image, "B11")
    red = _select_band(image, "B4")
    result = _safe_divide(swir1.subtract(red), swir1.add(red))
    return _finish(result, "NDSI_Soil", geometry, scale_m)


def ndti(image: ee.Image, geometry: ee.Geometry, scale_m: int) -> ee.Image:
    swir1 = _select_band(image, "B11")
    swir2 = _select_band(image, "B12")
    result = _safe_divide(swir1.subtract(swir2), swir1.add(swir2))
    return _finish(result, "NDTI", geometry, scale_m)


def pri(image: ee.Image, geometry: ee.Geometry, scale_m: int) -> ee.Image:
    blue = _select_band(image, "B2")
    green = _select_band(image, "B3")
    result = _safe_divide(blue.subtract(green), blue.add(green))
    return _finish(result, "PRI", geometry, scale_m)


INDEX_BUILDERS: Dict[str, IndexFunction] = {
    "NDVI": ndvi,
    "EVI": evi,
    "GNDVI": gndvi,
    "NDRE": ndre,
    "SAVI": savi,
    "MSAVI": msavi,
    "VARI": vari,
    "MCARI": mcari,
    "NDWI_McFeeters": ndwi_mcfeeters,
    "NDWI_Gao": ndwi_gao,
    "NDMI": ndmi,
    "MSI": msi,
    "GVMI": gvmi,
    "NBR": nbr,
    "PSRI": psri,
    "ARI": ari,
    "CRI": cri,
    "BSI": bsi,
    "SBI": sbi,
    "NDSI_Soil": ndsi_soil,
    "NDTI": ndti,
    "PRI": pri,
}

SUPPORTED_INDICES = sorted(INDEX_BUILDERS.keys())


# ---- Local evaluators for unit tests ---------------------------------------------------------

ScalarBands = Mapping[str, float]
ScalarFunction = Callable[[ScalarBands], float]


def _scalar_safe_divide(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return math.nan
    return numerator / denominator


def _scalar_msavi(values: ScalarBands) -> float:
    nir = values["B8"]
    red = values["B4"]
    two_nir_plus_one = 2 * nir + 1
    radicand = two_nir_plus_one**2 - 8 * (nir - red)
    if radicand < 0:
        return math.nan
    return (two_nir_plus_one - math.sqrt(radicand)) / 2


SCALAR_BUILDERS: Dict[str, ScalarFunction] = {
    "NDVI": lambda v: _scalar_safe_divide(v["B8"] - v["B4"], v["B8"] + v["B4"]),
    "EVI": lambda v: _scalar_safe_divide(
        2.5 * (v["B8"] - v["B4"]), v["B8"] + 6 * v["B4"] - 7.5 * v["B2"] + 1
    ),
    "GNDVI": lambda v: _scalar_safe_divide(v["B8"] - v["B3"], v["B8"] + v["B3"]),
    "NDRE": lambda v: _scalar_safe_divide(v["B8"] - v["B5"], v["B8"] + v["B5"]),
    "SAVI": lambda v: 1.5
    * _scalar_safe_divide(v["B8"] - v["B4"], v["B8"] + v["B4"] + 0.5),
    "MSAVI": _scalar_msavi,
    "VARI": lambda v: _scalar_safe_divide(
        v["B3"] - v["B4"], v["B3"] + v["B4"] - v["B2"]
    ),
    "MCARI": lambda v: ((v["B5"] - v["B4"]) - 0.2 * (v["B5"] - v["B3"]))
    * _scalar_safe_divide(v["B5"], v["B4"]),
    "NDWI_McFeeters": lambda v: _scalar_safe_divide(
        v["B3"] - v["B8"], v["B3"] + v["B8"]
    ),
    "NDWI_Gao": lambda v: _scalar_safe_divide(v["B8"] - v["B11"], v["B8"] + v["B11"]),
    "NDMI": lambda v: _scalar_safe_divide(v["B8"] - v["B11"], v["B8"] + v["B11"]),
    "MSI": lambda v: _scalar_safe_divide(v["B11"], v["B8"]),
    "GVMI": lambda v: _scalar_safe_divide(
        (v["B8"] + 0.1) - (v["B11"] + 0.02), (v["B8"] + 0.1) + (v["B11"] + 0.02)
    ),
    "NBR": lambda v: _scalar_safe_divide(v["B8"] - v["B12"], v["B8"] + v["B12"]),
    "PSRI": lambda v: _scalar_safe_divide(v["B4"] - v["B3"], v["B6"]),
    "ARI": lambda v: _scalar_safe_divide(1.0, v["B3"])
    - _scalar_safe_divide(1.0, v["B5"]),
    "CRI": lambda v: _scalar_safe_divide(1.0, v["B2"])
    - _scalar_safe_divide(1.0, v["B5"]),
    "BSI": lambda v: _scalar_safe_divide(
        (v["B11"] + v["B4"]) - (v["B8"] + v["B2"]),
        (v["B11"] + v["B4"]) + (v["B8"] + v["B2"]),
    ),
    "SBI": lambda v: math.sqrt(max((v["B4"] ** 2 + v["B3"] ** 2) / 2, 0)),
    "NDSI_Soil": lambda v: _scalar_safe_divide(v["B11"] - v["B4"], v["B11"] + v["B4"]),
    "NDTI": lambda v: _scalar_safe_divide(v["B11"] - v["B12"], v["B11"] + v["B12"]),
    "PRI": lambda v: _scalar_safe_divide(v["B2"] - v["B3"], v["B2"] + v["B3"]),
}


def compute_index(
    image: ee.Image, name: str, geometry: ee.Geometry, scale_m: int
) -> ee.Image:
    builder = INDEX_BUILDERS.get(name)
    if builder is None:
        raise KeyError(f"Unsupported index: {name}")
    return builder(image, geometry, scale_m)


def compute_scalar_index(name: str, values: ScalarBands) -> float:
    builder = SCALAR_BUILDERS.get(name)
    if builder is None:
        raise KeyError(f"Unsupported index: {name}")
    return builder(values)
