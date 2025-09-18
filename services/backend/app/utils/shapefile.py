import io
import os
import tempfile
import zipfile
from typing import List

import shapefile  # pyshp
from fastapi import HTTPException
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.ops import unary_union


def as_multipolygon(geoms: List[Polygon | MultiPolygon]) -> MultiPolygon:
    polys: List[Polygon] = []
    for geom in geoms:
        if isinstance(geom, Polygon):
            polys.append(geom)
        elif isinstance(geom, MultiPolygon):
            polys.extend(list(geom.geoms))
    if not polys:
        raise HTTPException(status_code=400, detail="No polygon features found.")
    merged = unary_union(polys)
    if isinstance(merged, Polygon):
        return MultiPolygon([merged])
    if isinstance(merged, MultiPolygon):
        return merged
    raise HTTPException(status_code=400, detail="Could not merge polygons.")


def shapefile_zip_to_geojson(file_bytes: bytes) -> dict:
    """Convert a shapefile ZIP archive to a GeoJSON MultiPolygon."""

    with tempfile.TemporaryDirectory() as td:
        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes), "r") as zf:
                zf.extractall(td)
        except zipfile.BadZipFile as exc:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid ZIP archive.") from exc

        shp_path = None
        for name in os.listdir(td):
            if name.lower().endswith(".shp"):
                shp_path = os.path.join(td, name)
                break

        if not shp_path:
            raise HTTPException(status_code=400, detail="ZIP must contain a .shp file.")

        try:
            reader = shapefile.Reader(shp_path)
        except shapefile.ShapefileException as exc:
            raise HTTPException(status_code=400, detail=f"Could not read shapefile: {exc}") from exc

        geoms: List[Polygon | MultiPolygon] = []
        for shp in reader.shapes():
            try:
                geom = shape(shp.__geo_interface__)
            except Exception:
                continue
            if isinstance(geom, (Polygon, MultiPolygon)):
                geoms.append(geom)

        multipolygon = as_multipolygon(geoms)
        return mapping(multipolygon)
