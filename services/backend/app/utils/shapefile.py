import io
import logging
import os
import tempfile
import zipfile
from pathlib import Path
from typing import List, Tuple

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


def shapefile_zip_to_geojson(
    file_bytes: bytes, source_epsg: int | str | None = None
) -> Tuple[dict, bool]:
    """Convert a shapefile ZIP archive to a GeoJSON MultiPolygon.

    Returns a tuple of the GeoJSON mapping and a flag indicating whether the CRS
    was defaulted to EPSG:4326 due to missing projection information.
    """

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

        shp_file_path = Path(shp_path)
        prj_path: Path | None = None
        try:
            siblings = sorted(shp_file_path.parent.glob(f"{shp_file_path.stem}.*"))
        except FileNotFoundError:
            siblings = []
        for candidate in siblings:
            if candidate.suffix.lower() == ".prj" and candidate.is_file():
                prj_path = candidate
                break
        source_crs: CRS | None = None
        defaulted_to_wgs84 = False
        crs_wkt: str | None = None
        if prj_path is not None:
            try:
                crs_wkt = prj_path.read_text()
            except FileNotFoundError:
                crs_wkt = None
            except OSError as exc:
                raise HTTPException(
                    status_code=400,
                    detail="Could not read shapefile .prj file to determine the coordinate reference system.",
                ) from exc

        if crs_wkt is not None:
            try:
                source_crs = CRS.from_wkt(crs_wkt)
            except CRSError as exc:
                raise HTTPException(
                    status_code=400,
                    detail="Shapefile .prj file does not define a valid coordinate reference system.",
                ) from exc
        elif source_epsg is not None:
            try:
                epsg_code = int(str(source_epsg).strip())
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=400,
                    detail="Provided EPSG code must be a valid integer.",
                ) from exc
            try:
                source_crs = CRS.from_epsg(epsg_code)
            except CRSError as exc:
                raise HTTPException(
                    status_code=400,
                    detail="Provided EPSG code does not define a valid coordinate reference system.",
                ) from exc
        else:
            source_crs = CRS.from_epsg(4326)
            defaulted_to_wgs84 = True
            logger.warning(
                "Shapefile missing .prj file and source_epsg; defaulting CRS to EPSG:4326 (WGS84)."
            )

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
        return mapping(multipolygon), defaulted_to_wgs84
