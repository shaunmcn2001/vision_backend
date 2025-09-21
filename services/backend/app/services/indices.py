"""Index definitions and helpers for Sentinel-2 based vegetation metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Tuple

import ee


class UnsupportedIndexError(ValueError):
    """Raised when an unknown vegetation index is requested."""


ParameterBuilder = Callable[[Mapping[str, Any] | None], Dict[str, Any]]
IndexComputer = Callable[[ee.Image, Mapping[str, Any]], ee.Image]


def _default_parameter_builder(params: Mapping[str, Any] | None) -> Dict[str, Any]:
    return dict(params or {})


@dataclass(frozen=True)
class IndexDefinition:
    code: str
    band_name: str
    valid_range: Tuple[float, float] | None
    compute: IndexComputer
    default_palette: Tuple[str, ...] | None = None
    parameter_builder: ParameterBuilder = field(default=_default_parameter_builder)

    def prepare_parameters(self, params: Mapping[str, Any] | None = None) -> Dict[str, Any]:
        return self.parameter_builder(params)

    def default_visualization(self) -> Dict[str, Any]:
        vis: Dict[str, Any] = {"bands": [self.band_name]}
        if self.valid_range is not None:
            vis["min"], vis["max"] = self.valid_range
        if self.default_palette is not None:
            vis["palette"] = list(self.default_palette)
        return vis


def _normalized_difference(band_pair: Tuple[str, str], name: str) -> IndexComputer:
    def _compute(image: ee.Image, params: Mapping[str, Any]) -> ee.Image:
        return image.normalizedDifference(list(band_pair)).rename(name)

    return _compute


def _compute_evi(image: ee.Image, params: Mapping[str, Any]) -> ee.Image:
    nir = image.select("B8")
    red = image.select("B4")
    blue = image.select("B2")
    numerator = nir.subtract(red).multiply(2.5)
    denominator = (
        nir.add(red.multiply(6))
        .subtract(blue.multiply(7.5))
        .add(ee.Image.constant(1))
    )
    return numerator.divide(denominator).rename("EVI")


def _prepare_ndre_parameters(params: Mapping[str, Any] | None) -> Dict[str, Any]:
    data = dict(params or {})
    nir_band = str(data.get("nir_band", "B8")).upper()
    if nir_band not in {"B8", "B8A"}:
        raise ValueError("NDRE supports nir_band 'B8' or 'B8A'.")
    data["nir_band"] = nir_band

    red_edge_band = str(data.get("red_edge_band", "B5")).upper()
    if red_edge_band not in {"B5", "B6", "B7"}:
        raise ValueError("NDRE red_edge_band must be one of B5, B6, or B7.")
    data["red_edge_band"] = red_edge_band
    return data


def _compute_ndre(image: ee.Image, params: Mapping[str, Any]) -> ee.Image:
    bands = [params["nir_band"], params["red_edge_band"]]
    return image.normalizedDifference(bands).rename("NDRE")


SUPPORTED_INDICES: Tuple[str, ...] = ("ndvi", "evi", "gndvi", "ndre")


DEFAULT_VEGETATION_PALETTE: Tuple[str, ...] = (
    "440154",
    "482173",
    "433E85",
    "38598C",
    "2D708E",
    "25858E",
    "1E9B8A",
    "2BB07F",
    "51C56A",
    "85D54A",
    "C2DF23",
    "FDE725",
)


INDEX_DEFINITIONS: Dict[str, IndexDefinition] = {
    "ndvi": IndexDefinition(
        code="ndvi",
        band_name="NDVI",
        valid_range=(-1.0, 1.0),
        compute=_normalized_difference(("B8", "B4"), "NDVI"),
        default_palette=DEFAULT_VEGETATION_PALETTE,
    ),
    "evi": IndexDefinition(
        code="evi",
        band_name="EVI",
        valid_range=(-1.0, 1.0),
        compute=_compute_evi,
        default_palette=DEFAULT_VEGETATION_PALETTE,
    ),
    "gndvi": IndexDefinition(
        code="gndvi",
        band_name="GNDVI",
        valid_range=(-1.0, 1.0),
        compute=_normalized_difference(("B8", "B3"), "GNDVI"),
        default_palette=DEFAULT_VEGETATION_PALETTE,
    ),
    "ndre": IndexDefinition(
        code="ndre",
        band_name="NDRE",
        valid_range=(-1.0, 1.0),
        compute=_compute_ndre,
        parameter_builder=_prepare_ndre_parameters,
        default_palette=DEFAULT_VEGETATION_PALETTE,
    ),
}


def normalize_index_code(code: str) -> str:
    definition = get_index_definition(code)
    return definition.code


def get_index_definition(code: str) -> IndexDefinition:
    try:
        return INDEX_DEFINITIONS[code.lower()]
    except KeyError as exc:  # pragma: no cover - defensive
        raise UnsupportedIndexError(f"Unsupported index '{code}'.") from exc


def resolve_index(
    code: str, params: Mapping[str, Any] | None = None
) -> tuple[IndexDefinition, Dict[str, Any]]:
    definition = get_index_definition(code)
    prepared = definition.prepare_parameters(params)
    return definition, prepared


__all__ = [
    "INDEX_DEFINITIONS",
    "IndexDefinition",
    "SUPPORTED_INDICES",
    "UnsupportedIndexError",
    "get_index_definition",
    "normalize_index_code",
    "resolve_index",
]
