import io
import logging
import os
import tempfile
import zipfile
from pathlib import Path
from typing import List

import shapefile  # pyshp
from fastapi import HTTPException
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.ops import transform, unary_union

from pyproj import CRS, Transformer
from pyproj.exceptions import CRSError

logger = logging.getLogger(__name__)


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
        for root, _, files in os.walk(td):
            for filename in files:
                if filename.lower().endswith(".shp"):
                    shp_path = os.path.join(root, filename)
                    break
            if shp_path:
                break

        if not shp_path:
            logger.warning("ZIP archive contained no .shp file after recursive search.")
            raise HTTPException(status_code=400, detail="ZIP archive must contain a polygon .shp file.")

        logger.debug("Using shapefile located at %s", shp_path)

        try:
            reader = shapefile.Reader(shp_path)
        except shapefile.ShapefileException as exc:
            raise HTTPException(status_code=400, detail=f"Could not read shapefile: {exc}") from exc

        prj_path = Path(shp_path).with_suffix(".prj")
        try:
            crs_wkt = prj_path.read_text()
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=400,
                detail="Shapefile must include a .prj file specifying the coordinate reference system.",
            ) from exc
        except OSError as exc:
            raise HTTPException(
                status_code=400,
                detail="Could not read shapefile .prj file to determine the coordinate reference system.",
            ) from exc

        try:
            source_crs = CRS.from_wkt(crs_wkt)
        except CRSError as exc:
            raise HTTPException(
                status_code=400,
                detail="Shapefile .prj file does not define a valid coordinate reference system.",
            ) from exc

        target_crs = CRS.from_epsg(4326)
        transformer: Transformer | None = None
        if not source_crs.equals(target_crs):
            try:
                transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
            except Exception as exc:  # pragma: no cover - defensive, should not occur with valid CRS
                raise HTTPException(
                    status_code=400,
                    detail="Could not transform shapefile geometries to WGS84 coordinate system.",
                ) from exc

        geoms: List[Polygon | MultiPolygon] = []
        for shp in reader.shapes():
            try:
                geom = shape(shp.__geo_interface__)
            except Exception:
                continue
            if isinstance(geom, (Polygon, MultiPolygon)):
                if transformer is not None:
                    geom = transform(transformer.transform, geom)
                geoms.append(geom)

        multipolygon = as_multipolygon(geoms)
        return mapping(multipolygon)
