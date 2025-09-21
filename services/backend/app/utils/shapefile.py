import io
import logging
import os
import shutil
import tempfile
import zipfile
from pathlib import PurePosixPath
from typing import List

import shapefile  # pyshp
from fastapi import HTTPException
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.ops import unary_union

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
                base_path = os.path.abspath(td)
                for member in zf.infolist():
                    member_name = member.filename
                    if not member_name:
                        continue

                    normalized_name = member_name.replace("\\", "/")
                    path_parts = PurePosixPath(normalized_name).parts
                    if any(part == ".." for part in path_parts):
                        raise HTTPException(
                            status_code=400,
                            detail="ZIP archive contains unsafe member paths.",
                        )

                    member_target = os.path.abspath(
                        os.path.join(base_path, *path_parts)
                    )
                    if not (
                        member_target == base_path
                        or member_target.startswith(base_path + os.sep)
                    ):
                        raise HTTPException(
                            status_code=400,
                            detail="ZIP archive contains unsafe member paths.",
                        )

                    if member.is_dir():
                        os.makedirs(member_target, exist_ok=True)
                        continue

                    os.makedirs(os.path.dirname(member_target), exist_ok=True)
                    with zf.open(member, "r") as source, open(
                        member_target, "wb"
                    ) as target:
                        shutil.copyfileobj(source, target)
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
