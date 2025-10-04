import csv
from pathlib import Path
import sys

TEST_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TEST_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import numpy as np
import rasterio
from rasterio.transform import from_origin
import shapefile

from app.services import zones


def _write_ndvi_raster(path: Path) -> None:
    data = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, -9999.0],
        ],
        dtype=np.float32,
    )
    transform = from_origin(0, 30, zones.DEFAULT_SCALE, zones.DEFAULT_SCALE)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs=zones.DEFAULT_EXPORT_CRS,
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)


def test_classify_local_zones_generates_outputs(tmp_path: Path) -> None:
    ndvi_path = tmp_path / "mean_ndvi.tif"
    _write_ndvi_raster(ndvi_path)

    artifacts, metadata = zones._classify_local_zones(
        ndvi_path,
        working_dir=tmp_path,
        n_classes=4,
        min_mapping_unit_ha=0.0,
        smooth_radius_m=0,
        open_radius_m=0,
        close_radius_m=0,
        include_stats=True,
    )

    assert Path(artifacts.raster_path).exists()
    assert Path(artifacts.vector_path).exists()
    assert artifacts.vector_components
    assert artifacts.zonal_stats_path is not None
    assert Path(artifacts.zonal_stats_path).exists()

    with rasterio.open(artifacts.raster_path) as classified:
        classes = np.unique(classified.read(1))
    assert set(classes) >= {0, 1, 2, 3, 4}

    assert "percentile_thresholds" in metadata
    assert len(metadata["percentile_thresholds"]) == 3
    assert metadata["zones"]

    with open(artifacts.zonal_stats_path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
    assert any(float(row["mean_ndvi"]) > 0 for row in rows)


def test_export_selected_period_zones_returns_local_paths(monkeypatch, tmp_path: Path) -> None:
    def fake_prepare(aoi_geojson, **kwargs):
        workdir = kwargs["working_dir"]
        ndvi_path = workdir / "mean_ndvi.tif"
        _write_ndvi_raster(ndvi_path)
        artifacts, local_meta = zones._classify_local_zones(
            ndvi_path,
            working_dir=workdir,
            n_classes=kwargs["n_classes"],
            min_mapping_unit_ha=kwargs["min_mapping_unit_ha"],
            smooth_radius_m=kwargs["smooth_radius_m"],
            open_radius_m=kwargs["open_radius_m"],
            close_radius_m=kwargs["close_radius_m"],
            include_stats=kwargs["include_stats"],
        )
        metadata = dict(local_meta)
        metadata.update({"used_months": list(kwargs["months"]), "skipped_months": []})
        return artifacts, metadata

    monkeypatch.setattr(zones, "_prepare_selected_period_artifacts", fake_prepare)
    monkeypatch.setattr(zones.gee, "initialize", lambda: None)
    monkeypatch.setattr(zones, "_to_ee_geometry", lambda geom: geom)
    monkeypatch.setattr(zones, "_resolve_geometry", lambda geom: geom)

    aoi = {
        "type": "Polygon",
        "coordinates": [
            [
                [0.0, 0.0],
                [0.0, 0.001],
                [0.001, 0.001],
                [0.001, 0.0],
                [0.0, 0.0],
            ]
        ],
    }

    result = zones.export_selected_period_zones(
        aoi,
        "test-aoi",
        ["2024-01"],
        export_target="local",
    )

    paths = result["paths"]
    assert Path(paths["raster"]).exists()
    assert Path(paths["vectors"]).exists()
    if paths["zonal_stats"] is not None:
        assert Path(paths["zonal_stats"]).exists()

    assert result["metadata"]["used_months"] == ["2024-01"]
    assert result["prefix"].startswith("zones/PROD_202401")
    assert Path(result["working_dir"]).exists()
    assert result["tasks"] == {}

    reader = shapefile.Reader(str(paths["vectors"]))
    field_names = [field[0] for field in reader.fields[1:]]
    zone_index = field_names.index("zone") if "zone" in field_names else 0
    zones_in_file = {
        record.record[zone_index]
        for record in reader.iterShapeRecords()
    }
    assert set(zones_in_file) <= set(range(1, 6))
