import sys
import zipfile
from pathlib import Path

import pytest
import shapefile  # pyshp
from fastapi import HTTPException
from pyproj import CRS, Transformer
from shapely.geometry import shape

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

from app.utils.shapefile import shapefile_zip_to_geojson


def _write_polygon_shapefile(base_path: Path, coords):
    writer = shapefile.Writer(str(base_path))
    writer.field("id", "N")
    writer.poly([coords])
    writer.record(1)
    writer.close()


def test_shapefile_zip_to_geojson_transforms_epsg3857_to_wgs84(tmp_path):
    wgs84_coords = [
        (-122.0, 37.0),
        (-122.0, 37.0005),
        (-121.9995, 37.0005),
        (-121.9995, 37.0),
        (-122.0, 37.0),
    ]

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    projected_coords = [transformer.transform(lon, lat) for lon, lat in wgs84_coords]

    base_path = tmp_path / "test_polygon"
    _write_polygon_shapefile(base_path, projected_coords)

    prj_path = base_path.with_suffix(".prj")
    prj_path.write_text(CRS.from_epsg(3857).to_wkt())

    zip_path = tmp_path / "polygon.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for ext in ("shp", "shx", "dbf", "prj"):
            zf.write(str(base_path.with_suffix(f".{ext}")), arcname=f"test_polygon.{ext}")

    geojson = shapefile_zip_to_geojson(zip_path.read_bytes())

    geom = shape(geojson)
    assert geom.geom_type == "MultiPolygon"
    polygon = list(geom.geoms)[0]
    converted_coords = list(polygon.exterior.coords)

    assert len(converted_coords) == len(wgs84_coords)
    for (expected_lon, expected_lat), (lon, lat) in zip(wgs84_coords, converted_coords):
        assert lon == pytest.approx(expected_lon, abs=1e-6)
        assert lat == pytest.approx(expected_lat, abs=1e-6)


def test_shapefile_zip_to_geojson_missing_prj_raises(tmp_path):
    wgs84_coords = [
        (-122.0, 37.0),
        (-122.0, 37.0005),
        (-121.9995, 37.0005),
        (-121.9995, 37.0),
        (-122.0, 37.0),
    ]

    base_path = tmp_path / "test_polygon_missing_prj"
    _write_polygon_shapefile(base_path, wgs84_coords)

    zip_path = tmp_path / "polygon_missing_prj.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for ext in ("shp", "shx", "dbf"):
            zf.write(str(base_path.with_suffix(f".{ext}")), arcname=f"test_polygon_missing_prj.{ext}")

    with pytest.raises(HTTPException) as excinfo:
        shapefile_zip_to_geojson(zip_path.read_bytes())

    assert excinfo.value.status_code == 400
    assert ".prj" in excinfo.value.detail
