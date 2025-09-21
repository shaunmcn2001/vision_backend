import sys
import zipfile
from pathlib import Path

import pytest
import shapefile  # pyshp
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

    geojson, warnings = shapefile_zip_to_geojson(zip_path.read_bytes())

    assert warnings == []

    geom = shape(geojson)
    assert geom.geom_type == "MultiPolygon"
    polygon = list(geom.geoms)[0]
    converted_coords = list(polygon.exterior.coords)

    assert len(converted_coords) == len(wgs84_coords)
    for (expected_lon, expected_lat), (lon, lat) in zip(wgs84_coords, converted_coords):
        assert lon == pytest.approx(expected_lon, abs=1e-6)
        assert lat == pytest.approx(expected_lat, abs=1e-6)


def test_shapefile_zip_to_geojson_honors_uppercase_prj(tmp_path):
    wgs84_coords = [
        (-122.0, 37.0),
        (-122.0, 37.0005),
        (-121.9995, 37.0005),
        (-121.9995, 37.0),
        (-122.0, 37.0),
    ]

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    projected_coords = [transformer.transform(lon, lat) for lon, lat in wgs84_coords]

    base_path = tmp_path / "test_polygon_uppercase_prj"
    _write_polygon_shapefile(base_path, projected_coords)

    prj_path = base_path.with_suffix(".PRJ")
    prj_path.write_text(CRS.from_epsg(3857).to_wkt())

    zip_path = tmp_path / "polygon_uppercase_prj.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for ext in ("shp", "shx", "dbf"):
            zf.write(
                str(base_path.with_suffix(f".{ext}")),
                arcname=f"test_polygon_uppercase_prj.{ext}",
            )
        zf.write(str(prj_path), arcname="test_polygon_uppercase_prj.PRJ")

    geojson, warnings = shapefile_zip_to_geojson(zip_path.read_bytes())

    assert warnings == []

    geom = shape(geojson)
    assert geom.geom_type == "MultiPolygon"
    polygon = list(geom.geoms)[0]
    converted_coords = list(polygon.exterior.coords)

    assert len(converted_coords) == len(wgs84_coords)
    for (expected_lon, expected_lat), (lon, lat) in zip(wgs84_coords, converted_coords):
        assert lon == pytest.approx(expected_lon, abs=1e-6)
        assert lat == pytest.approx(expected_lat, abs=1e-6)


def test_shapefile_zip_to_geojson_missing_prj_geographic_heuristic(tmp_path):
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

    geojson, warnings = shapefile_zip_to_geojson(zip_path.read_bytes())

    expected_warning = (
        "Shapefile ZIP is missing projection information; heuristically assumed EPSG:4326 based on geographic coordinate extents."
        " Provide the .prj file in the ZIP or pass source_epsg=<EPSG code> to avoid this assumption."
    )
    assert warnings == [expected_warning]

    geom = shape(geojson)
    assert geom.geom_type == "MultiPolygon"
    polygon = list(geom.geoms)[0]
    converted_coords = list(polygon.exterior.coords)

    assert len(converted_coords) == len(wgs84_coords)
    for (expected_lon, expected_lat), (lon, lat) in zip(wgs84_coords, converted_coords):
        assert lon == pytest.approx(expected_lon, abs=1e-6)
        assert lat == pytest.approx(expected_lat, abs=1e-6)


def test_shapefile_zip_to_geojson_missing_prj_projected_heuristic(tmp_path):
    wgs84_coords = [
        (144.95, -37.82),
        (144.95, -37.81),
        (144.96, -37.81),
        (144.96, -37.82),
        (144.95, -37.82),
    ]

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:28355", always_xy=True)
    projected_coords = [transformer.transform(lon, lat) for lon, lat in wgs84_coords]

    base_path = tmp_path / "test_polygon_projected_missing_prj"
    _write_polygon_shapefile(base_path, projected_coords)

    zip_path = tmp_path / "polygon_projected_missing_prj.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for ext in ("shp", "shx", "dbf"):
            zf.write(
                str(base_path.with_suffix(f".{ext}")),
                arcname=f"test_polygon_projected_missing_prj.{ext}",
            )

    geojson, warnings = shapefile_zip_to_geojson(zip_path.read_bytes())

    epsg_name = CRS.from_epsg(28355).name
    expected_warning = (
        f"Shapefile ZIP is missing projection information; heuristically assumed EPSG:28355 ({epsg_name}) based on Australian bounds."
        " Provide the .prj file in the ZIP or pass source_epsg=<EPSG code> to avoid this assumption."
    )
    assert warnings == [expected_warning]

    geom = shape(geojson)
    assert geom.geom_type == "MultiPolygon"
    polygon = list(geom.geoms)[0]
    converted_coords = list(polygon.exterior.coords)

    assert len(converted_coords) == len(wgs84_coords)
    for (expected_lon, expected_lat), (lon, lat) in zip(wgs84_coords, converted_coords):
        assert lon == pytest.approx(expected_lon, abs=1e-6)
        assert lat == pytest.approx(expected_lat, abs=1e-6)


def test_shapefile_zip_to_geojson_missing_prj_with_epsg(tmp_path):
    wgs84_coords = [
        (-122.0, 37.0),
        (-122.0, 37.0005),
        (-121.9995, 37.0005),
        (-121.9995, 37.0),
        (-122.0, 37.0),
    ]

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    projected_coords = [transformer.transform(lon, lat) for lon, lat in wgs84_coords]

    base_path = tmp_path / "test_polygon_missing_prj_epsg"
    _write_polygon_shapefile(base_path, projected_coords)

    zip_path = tmp_path / "polygon_missing_prj_epsg.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for ext in ("shp", "shx", "dbf"):
            zf.write(
                str(base_path.with_suffix(f".{ext}")),
                arcname=f"test_polygon_missing_prj_epsg.{ext}",
            )

    geojson, warnings = shapefile_zip_to_geojson(zip_path.read_bytes(), source_epsg=3857)

    assert warnings == []

    geom = shape(geojson)
    assert geom.geom_type == "MultiPolygon"
    polygon = list(geom.geoms)[0]
    converted_coords = list(polygon.exterior.coords)

    assert len(converted_coords) == len(wgs84_coords)
    for (expected_lon, expected_lat), (lon, lat) in zip(wgs84_coords, converted_coords):
        assert lon == pytest.approx(expected_lon, abs=1e-6)
        assert lat == pytest.approx(expected_lat, abs=1e-6)


def test_shapefile_zip_to_geojson_missing_prj_without_inference(tmp_path, caplog):
    ambiguous_coords = [
        (500.0, 500.0),
        (500.0, 510.0),
        (510.0, 510.0),
        (510.0, 500.0),
        (500.0, 500.0),
    ]

    base_path = tmp_path / "test_polygon_missing_metadata"
    _write_polygon_shapefile(base_path, ambiguous_coords)

    zip_path = tmp_path / "polygon_missing_metadata.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for ext in ("shp", "shx", "dbf"):
            zf.write(str(base_path.with_suffix(f".{ext}")), arcname=f"test_polygon_missing_metadata.{ext}")

    with caplog.at_level("WARNING"):
        geojson, warnings = shapefile_zip_to_geojson(zip_path.read_bytes())

    assert len(warnings) == 1
    warning = warnings[0]
    expected_prefix = (
        "Missing CRS (.prj) and no source_epsg provided. Defaulting to EPSG:4326 (WGS84)."
        " Include the .prj in the ZIP or pass source_epsg=<EPSG code> to avoid this assumption."
    )
    assert warning.startswith(expected_prefix)
    assert "Detector hint" in warning
    assert "projected/metre-like" in warning
    assert "Observed bounds: [500.00, 500.00]–[510.00, 510.00]" in warning
    assert any(record.message == warning for record in caplog.records)

    geom = shape(geojson)
    assert geom.geom_type == "MultiPolygon"
    polygon = list(geom.geoms)[0]
    converted_coords = list(polygon.exterior.coords)

    assert len(converted_coords) == len(ambiguous_coords)
    for (expected_lon, expected_lat), (lon, lat) in zip(ambiguous_coords, converted_coords):
        assert lon == pytest.approx(expected_lon, abs=1e-6)
        assert lat == pytest.approx(expected_lat, abs=1e-6)
