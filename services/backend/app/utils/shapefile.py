import io
import logging
import math
import os
import tempfile
import zipfile
from pathlib import Path
from typing import List, Sequence, Tuple

import shapefile  # pyshp
from fastapi import HTTPException
from shapely.geometry import MultiPolygon, Point, Polygon, mapping, shape
from shapely.ops import transform, unary_union

from pyproj import CRS, Transformer
from pyproj.exceptions import CRSError

logger = logging.getLogger(__name__)

WGS84_CRS = CRS.from_epsg(4326)
AU_BOUNDS_WGS84 = (93.0, -60.0, 174.0, -7.0)
AU_CENTROID_WGS84 = (
    (AU_BOUNDS_WGS84[0] + AU_BOUNDS_WGS84[2]) / 2,
    (AU_BOUNDS_WGS84[1] + AU_BOUNDS_WGS84[3]) / 2,
)
LIKELY_AUSTRALIAN_EPSGS: Tuple[int, ...] = (
    28348,
    28349,
    28350,
    28351,
    28352,
    28353,
    28354,
    28355,
    28356,
    28357,
    28358,
    32748,
    32749,
    32750,
    32751,
    32752,
    32753,
    32754,
    32755,
    32756,
    32757,
    32758,
    3577,
    7845,
    7846,
    7847,
    7848,
    7849,
    7850,
    7851,
    7852,
    7853,
    7854,
    7855,
    7856,
    7857,
    7858,
    9473,
    3112,
)

AU_MAINLAND_APPROX = Polygon(
    [
        (112.0, -44.0),
        (114.0, -33.0),
        (124.0, -16.0),
        (130.5, -11.0),
        (138.0, -10.0),
        (154.0, -8.0),
        (154.0, -37.0),
        (148.0, -43.0),
        (140.0, -43.5),
        (136.0, -40.0),
        (132.0, -36.5),
        (127.0, -35.0),
        (120.0, -36.5),
        (116.0, -36.5),
        (112.0, -44.0),
    ]
)


def _candidate_bonus(epsg: int) -> float:
    if 28348 <= epsg <= 28358:
        return 0.08
    if 7845 <= epsg <= 7858:
        return 0.07
    if epsg in {3577, 9473, 3112}:
        return 0.05
    return 0.0


def _combined_bounds(shapes: Sequence[shapefile.Shape]) -> Tuple[float, float, float, float] | None:
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


def _looks_geographic(bounds: Tuple[float, float, float, float]) -> bool:
    minx, miny, maxx, maxy = bounds
    max_abs_lon = max(abs(minx), abs(maxx))
    max_abs_lat = max(abs(miny), abs(maxy))
    lon_span = maxx - minx
    lat_span = maxy - miny
    return (
        max_abs_lon <= 200
        and max_abs_lat <= 100
        and lon_span <= 360
        and lat_span <= 180
    )


def _bounds_area(bounds: Tuple[float, float, float, float]) -> float:
    minx, miny, maxx, maxy = bounds
    width = maxx - minx
    height = maxy - miny
    if width <= 0 or height <= 0:
        return 0.0
    return width * height


def _intersection_area(
    first: Tuple[float, float, float, float],
    second: Tuple[float, float, float, float],
) -> float:
    x_overlap = min(first[2], second[2]) - max(first[0], second[0])
    y_overlap = min(first[3], second[3]) - max(first[1], second[1])
    if x_overlap <= 0 or y_overlap <= 0:
        return 0.0
    return x_overlap * y_overlap


def _center_distance(bounds: Tuple[float, float, float, float], reference: Tuple[float, float]) -> float:
    cx = (bounds[0] + bounds[2]) / 2.0
    cy = (bounds[1] + bounds[3]) / 2.0
    return math.hypot(cx - reference[0], cy - reference[1])


def _transform_bounds(
    transformer: Transformer, bounds: Tuple[float, float, float, float]
) -> Tuple[float, float, float, float] | None:
    corners = (
        (bounds[0], bounds[1]),
        (bounds[0], bounds[3]),
        (bounds[2], bounds[1]),
        (bounds[2], bounds[3]),
    )
    transformed: List[Tuple[float, float]] = []
    for x, y in corners:
        try:
            tx, ty = transformer.transform(x, y)
        except Exception:  # pragma: no cover - defensive fallback
            return None
        if not (math.isfinite(tx) and math.isfinite(ty)):
            return None
        transformed.append((tx, ty))
    xs = [pt[0] for pt in transformed]
    ys = [pt[1] for pt in transformed]
    return (min(xs), min(ys), max(xs), max(ys))


def _score_candidate(
    bounds: Tuple[float, float, float, float], candidate_crs: CRS
) -> float | None:
    try:
        transformer = Transformer.from_crs(candidate_crs, WGS84_CRS, always_xy=True)
    except Exception:  # pragma: no cover - defensive fallback
        return None
    transformed_bounds = _transform_bounds(transformer, bounds)
    if transformed_bounds is None:
        return None
    area = _bounds_area(transformed_bounds)
    if area <= 0:
        return None
    intersection = _intersection_area(transformed_bounds, AU_BOUNDS_WGS84)
    if intersection <= 0:
        return 0.0
    overlap_ratio = intersection / area
    center_penalty = _center_distance(transformed_bounds, AU_CENTROID_WGS84) / 100.0
    cx = (transformed_bounds[0] + transformed_bounds[2]) / 2.0
    cy = (transformed_bounds[1] + transformed_bounds[3]) / 2.0
    centroid_point = Point(cx, cy)
    outside_distance = AU_MAINLAND_APPROX.distance(centroid_point)
    outside_penalty = min(outside_distance * 2.0, 1.0)
    interior_bonus = 0.0
    if outside_penalty == 0.0:
        interior_distance = AU_MAINLAND_APPROX.boundary.distance(centroid_point)
        interior_bonus = min(interior_distance / 15.0, 0.3)
    score = overlap_ratio - center_penalty - outside_penalty + interior_bonus
    return max(score, 0.0)


def _infer_missing_crs(
    shapes: Sequence[shapefile.Shape],
) -> Tuple[CRS | None, str | None]:
    bounds = _combined_bounds(shapes)
    if bounds is None:
        return None, None

    if _looks_geographic(bounds):
        message = (
            "Shapefile ZIP is missing projection information; heuristically assumed EPSG:4326 based on geographic coordinate extents."
            " Provide the .prj file in the ZIP or pass source_epsg=<EPSG code> to avoid this assumption."
        )
        return WGS84_CRS, message

    max_abs_value = max(abs(value) for value in bounds)
    if max_abs_value < 1000:
        return None, None

    best_score = 0.0
    best_raw_score = 0.0
    best_epsg: int | None = None
    best_crs: CRS | None = None
    for epsg in LIKELY_AUSTRALIAN_EPSGS:
        try:
            candidate_crs = CRS.from_epsg(epsg)
        except CRSError:  # pragma: no cover - handled by curated list
            continue
        score = _score_candidate(bounds, candidate_crs)
        if score is None:
            continue
        adjusted_score = score + _candidate_bonus(epsg)
        if adjusted_score > best_score:
            best_score = adjusted_score
            best_raw_score = score
            best_epsg = epsg
            best_crs = candidate_crs

    if best_epsg is not None and best_crs is not None and best_raw_score > 0:
        message = (
            f"Shapefile ZIP is missing projection information; heuristically assumed EPSG:{best_epsg} ({best_crs.name}) based on Australian bounds."
            " Provide the .prj file in the ZIP or pass source_epsg=<EPSG code> to avoid this assumption."
        )
        return best_crs, message

    return None, None


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
) -> Tuple[dict, List[str]]:
    """Convert a shapefile ZIP archive to a GeoJSON MultiPolygon and surface warnings."""

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
        warnings: List[str] = []
        crs_wkt: str | None = None
        shapes: List[shapefile.Shape] = reader.shapes()
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
            inferred_crs, heuristic_warning = _infer_missing_crs(shapes)
            if inferred_crs is not None:
                source_crs = inferred_crs
                if heuristic_warning:
                    warnings.append(heuristic_warning)
                    logger.warning(heuristic_warning)
            else:
                message = (
                    "Missing CRS (.prj) and no source_epsg provided. Include the .prj in the ZIP or pass source_epsg=<EPSG code>."
                )
                logger.warning(message)
                raise HTTPException(status_code=400, detail=message)

        target_crs = WGS84_CRS
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
        for shp in shapes:
            try:
                geom = shape(shp.__geo_interface__)
            except Exception:
                continue
            if isinstance(geom, (Polygon, MultiPolygon)):
                if transformer is not None:
                    geom = transform(transformer.transform, geom)
                geoms.append(geom)

        multipolygon = as_multipolygon(geoms)
        return mapping(multipolygon), warnings
