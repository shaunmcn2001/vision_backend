"""Heuristics that describe the coordinates stored in shapefiles without CRS metadata."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence, Tuple

import shapefile  # pyshp

Bounds = Tuple[float, float, float, float]
SuggestionCategory = Literal["geographic", "projected", "ambiguous"]


def _confidence_label(value: float) -> str:
    if value >= 0.85:
        return "high confidence"
    if value >= 0.6:
        return "medium confidence"
    return "low confidence"


@dataclass(frozen=True)
class CoordinateSystemSuggestion:
    """Structured output describing how coordinate ranges should be interpreted."""

    category: SuggestionCategory
    confidence: float
    bounds: Bounds
    rationale: str

    @property
    def human_readable_hint(self) -> str:
        label = _confidence_label(self.confidence)
        minx, miny, maxx, maxy = self.bounds
        descriptor: str
        if self.category == "geographic":
            descriptor = "coordinates appear to be geographic (degree-like) values"
        elif self.category == "projected":
            descriptor = "coordinates appear to be projected/metre-like values"
        else:
            descriptor = "coordinate ranges are ambiguous"
        details = (
            f"Detector hint ({label}): {descriptor} {self.rationale}."
            f" Observed bounds: [{minx:.2f}, {miny:.2f}]–[{maxx:.2f}, {maxy:.2f}]."
        )
        return details


def _collect_bounds(shapes: Sequence[shapefile.Shape]) -> Bounds | None:
    minx = math.inf
    miny = math.inf
    maxx = -math.inf
    maxy = -math.inf
    has_bounds = False
    for shp in shapes:
        bbox = getattr(shp, "bbox", None)
        points = getattr(shp, "points", None)
        if not bbox or len(bbox) != 4 or not points:
            continue
        if not all(math.isfinite(value) for value in bbox):
            continue
        sx0, sy0, sx1, sy1 = bbox
        minx = min(minx, sx0)
        miny = min(miny, sy0)
        maxx = max(maxx, sx1)
        maxy = max(maxy, sy1)
        has_bounds = True
    if not has_bounds:
        return None
    return (minx, miny, maxx, maxy)


def detect_coordinate_system_suggestion(
    shapes: Sequence[shapefile.Shape],
) -> CoordinateSystemSuggestion | None:
    """Attempt to classify whether coordinates look geographic or projected."""

    bounds = _collect_bounds(shapes)
    if bounds is None:
        return None

    minx, miny, maxx, maxy = bounds
    width = max(maxx - minx, 0.0)
    height = max(maxy - miny, 0.0)
    max_abs_value = max(abs(minx), abs(miny), abs(maxx), abs(maxy))
    max_lon_like = max(abs(minx), abs(maxx))
    max_lat_like = max(abs(miny), abs(maxy))
    max_span = max(width, height)

    looks_geographic = (
        max_lon_like <= 200
        and max_lat_like <= 100
        and width <= 360
        and height <= 180
    )
    if looks_geographic:
        confidence = 0.85
        if max_lon_like <= 180 and max_lat_like <= 90:
            confidence = 0.95
        rationale = (
            "based on values that remain within typical longitude/latitude bounds"
        )
        return CoordinateSystemSuggestion(
            category="geographic",
            confidence=confidence,
            bounds=bounds,
            rationale=rationale,
        )

    projected_high_magnitude = max_abs_value >= 10000 or max_span >= 5000
    if projected_high_magnitude:
        rationale = (
            "because magnitudes reach tens of thousands, aligning with metre-based grids"
        )
        return CoordinateSystemSuggestion(
            category="projected",
            confidence=0.95,
            bounds=bounds,
            rationale=rationale,
        )

    projected_confident = max_abs_value >= 1000 or max_span >= 1000
    if projected_confident:
        rationale = (
            "because coordinates exceed typical degree ranges by orders of magnitude"
        )
        return CoordinateSystemSuggestion(
            category="projected",
            confidence=0.85,
            bounds=bounds,
            rationale=rationale,
        )

    projected_possible = max_abs_value > 200 or max_span > 200
    if projected_possible:
        rationale = (
            "because values surpass realistic longitude/latitude limits"
        )
        return CoordinateSystemSuggestion(
            category="projected",
            confidence=0.6,
            bounds=bounds,
            rationale=rationale,
        )

    rationale = "because there was insufficient variation to classify the coordinates"
    return CoordinateSystemSuggestion(
        category="ambiguous",
        confidence=0.3,
        bounds=bounds,
        rationale=rationale,
    )
