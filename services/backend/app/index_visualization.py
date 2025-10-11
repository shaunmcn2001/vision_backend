"""Helpers for colourising exported index GeoTIFFs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import ee
from app.services.ee_patches import apply_ee_runtime_patches
from app.services.ee_debug import debug_trace, debug_wrap  # noqa: F401

apply_ee_runtime_patches()


@dataclass(frozen=True)
class VisualizationSpec:
    """Defines percentile stretch and palette for an index."""

    default_min: float | None
    default_max: float | None
    palette: Tuple[str, ...]
    percentile_min: int = 2
    percentile_max: int = 98


class VisualizationError(Exception):
    """Raised when visualisation parameters cannot be resolved."""


_MAX_PIXELS = int(1e12)


def _safe_float(value: object | None, fallback: float | None) -> float | None:
    if value is None:
        return fallback
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _resolve_percentiles(
    image: ee.Image,
    band: str,
    geometry: ee.Geometry,
    scale_m: int,
    percentile_min: int,
    percentile_max: int,
) -> dict[str, float] | None:
    reducer = ee.Reducer.percentile([percentile_min, percentile_max])
    try:
        stats = (
            image.select(band)
            .reduceRegion(
                reducer=reducer,
                geometry=geometry,
                scale=scale_m,
                bestEffort=True,
                maxPixels=_MAX_PIXELS,
            )
            .getInfo()
        )
    except Exception:
        return None
    if not isinstance(stats, dict) or not stats:
        return None
    return stats


def _pick_bounds(
    stats: dict[str, float] | None,
    band: str,
    spec: VisualizationSpec,
) -> tuple[float, float] | None:
    min_value = spec.default_min
    max_value = spec.default_max

    if stats:
        min_key = f"{band}_p{spec.percentile_min}"
        max_key = f"{band}_p{spec.percentile_max}"
        min_value = _safe_float(stats.get(min_key), min_value)
        max_value = _safe_float(stats.get(max_key), max_value)

    if min_value is None or max_value is None:
        return None

    min_clamped = min_value
    max_clamped = max_value

    if spec.default_min is not None:
        min_clamped = max(min_value, spec.default_min)
    if spec.default_max is not None:
        max_clamped = min(max_value, spec.default_max)

    if max_clamped <= min_clamped:
        epsilon = abs(max_clamped) if max_clamped != 0 else 1.0
        max_clamped = min_clamped + epsilon

    return float(min_clamped), float(max_clamped)


def prepare_image_for_export(
    image: ee.Image,
    index_name: str,
    geometry: ee.Geometry,
    scale_m: int,
) -> tuple[ee.Image, bool]:
    """Return a possibly colourised version of ``image`` for export."""

    spec = INDEX_VISUALIZATION_SPECS.get(index_name)
    if spec is None:
        return image, False

    stats = _resolve_percentiles(
        image,
        index_name,
        geometry,
        scale_m,
        spec.percentile_min,
        spec.percentile_max,
    )
    bounds = _pick_bounds(stats, index_name, spec)
    if bounds is None:
        return image, False

    min_value, max_value = bounds
    palette: Iterable[str] = spec.palette

    visualized = image.visualize(
        min=min_value,
        max=max_value,
        palette=list(palette),
    )
    return visualized, True


# Palettes sourced from agronomic visualisation defaults.
_PALETTES: Dict[str, Tuple[str, ...]] = {
    "greenVeg": (
        "#704d2a",
        "#b98f5a",
        "#e2d7b5",
        "#a9d27e",
        "#5dbb63",
        "#2e8b57",
        "#166c3a",
    ),
    "chlor": (
        "#ffffcc",
        "#c7e9b4",
        "#7fcdbb",
        "#41b6c4",
        "#1d91c0",
        "#225ea8",
    ),
    "moistBlue": (
        "#f7fbff",
        "#deebf7",
        "#c6dbef",
        "#9ecae1",
        "#6baed6",
        "#3182bd",
        "#08519c",
    ),
    "dryWet": (
        "#fff5f0",
        "#fee0d2",
        "#fcbba1",
        "#fc9272",
        "#fb6a4a",
        "#de2d26",
        "#a50f15",
    ),
    "fire": (
        "#ffffe5",
        "#d9f0a3",
        "#addd8e",
        "#78c679",
        "#31a354",
        "#006837",
    ),
    "diverge1": (
        "#2166ac",
        "#67a9cf",
        "#d1e5f0",
        "#f7f7f7",
        "#fddbc7",
        "#ef8a62",
        "#b2182b",
    ),
    "cyanSeq": (
        "#f7fcf0",
        "#e0f3db",
        "#ccebc5",
        "#a8ddb5",
        "#7bccc4",
        "#4eb3d3",
        "#2b8cbe",
    ),
    "purpleSeq": (
        "#fff7fb",
        "#ece7f2",
        "#d0d1e6",
        "#a6bddb",
        "#74a9cf",
        "#2b8cbe",
        "#045a8d",
    ),
    "soil": (
        "#3d2b1f",
        "#6b4f2a",
        "#a6785a",
        "#d4b483",
        "#ead7c0",
    ),
    "gray": (
        "#2b2b2b",
        "#575757",
        "#8a8a8a",
        "#bcbcbc",
        "#e6e6e6",
    ),
    "soilGreen": (
        "#edf8fb",
        "#ccece6",
        "#99d8c9",
        "#66c2a4",
        "#2ca25f",
        "#006d2c",
    ),
    "till": (
        "#f7f4ea",
        "#ead9c3",
        "#d8b892",
        "#c6965f",
        "#9d6b3b",
        "#6b4423",
    ),
    "diverge2": (
        "#313695",
        "#74add1",
        "#e0f3f8",
        "#ffffbf",
        "#fdae61",
        "#d73027",
    ),
    "stress": (
        "#2c7bb6",
        "#abd9e9",
        "#ffffbf",
        "#fdae61",
        "#d7191c",
    ),
}


INDEX_VISUALIZATION_SPECS: Dict[str, VisualizationSpec] = {
    "NDVI": VisualizationSpec(-0.2, 0.9, _PALETTES["greenVeg"]),
    "EVI": VisualizationSpec(0.0, 0.8, _PALETTES["greenVeg"]),
    "GNDVI": VisualizationSpec(-0.2, 0.9, _PALETTES["greenVeg"]),
    "NDRE": VisualizationSpec(0.0, 0.5, _PALETTES["chlor"]),
    "SAVI": VisualizationSpec(0.0, 0.9, _PALETTES["greenVeg"]),
    "MSAVI": VisualizationSpec(0.0, 0.9, _PALETTES["greenVeg"]),
    "VARI": VisualizationSpec(-0.1, 0.6, _PALETTES["greenVeg"]),
    "MCARI": VisualizationSpec(0.0, 3.0, _PALETTES["greenVeg"]),
    "NDWI_McFeeters": VisualizationSpec(-0.5, 0.5, _PALETTES["moistBlue"]),
    "NDWI_Gao": VisualizationSpec(-0.2, 0.6, _PALETTES["dryWet"]),
    "NDMI": VisualizationSpec(-0.2, 0.6, _PALETTES["dryWet"]),
    "MSI": VisualizationSpec(0.3, 2.0, _PALETTES["stress"]),
    "GVMI": VisualizationSpec(-0.1, 0.8, _PALETTES["moistBlue"]),
    "NBR": VisualizationSpec(-0.1, 0.9, _PALETTES["fire"]),
    "PSRI": VisualizationSpec(-0.2, 0.6, _PALETTES["diverge1"]),
    "ARI": VisualizationSpec(0.0, 0.2, _PALETTES["cyanSeq"]),
    "CRI": VisualizationSpec(0.0, 1.0, _PALETTES["purpleSeq"]),
    "BSI": VisualizationSpec(-0.2, 0.6, _PALETTES["soil"]),
    "SBI": VisualizationSpec(0.05, 0.5, _PALETTES["gray"]),
    "NDSI_Soil": VisualizationSpec(-0.2, 0.6, _PALETTES["soilGreen"]),
    "NDTI": VisualizationSpec(-0.2, 0.6, _PALETTES["till"]),
    "PRI": VisualizationSpec(-0.2, 0.2, _PALETTES["diverge2"]),
}


__all__ = [
    "INDEX_VISUALIZATION_SPECS",
    "VisualizationError",
    "VisualizationSpec",
    "prepare_image_for_export",
]
