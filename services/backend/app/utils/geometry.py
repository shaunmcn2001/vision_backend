"""Geometry helpers shared across API endpoints."""
from __future__ import annotations

from shapely.geometry import shape
from shapely.ops import transform as shp_transform
from shapely.validation import make_valid
from pyproj import Transformer

# Australia Albers (equal-area). Matches the projection used for field areas.
_EQUAL_AREA_TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:3577", always_xy=True)


def area_m2(geojson_geom: dict) -> float:
    """Return the area of a GeoJSON geometry in square metres using an equal-area CRS."""
    geom = make_valid(shape(geojson_geom))
    reproj = shp_transform(lambda x, y, z=None: _EQUAL_AREA_TRANSFORMER.transform(x, y), geom)
    return float(reproj.area)


def area_ha(geojson_geom: dict) -> float:
    """Return the area of a GeoJSON geometry in hectares."""
    return area_m2(geojson_geom) / 10000.0
