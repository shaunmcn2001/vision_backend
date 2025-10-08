import csv
import io
import sys
import zipfile
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rasterio
import shapefile
from rasterio.transform import from_origin
from shapely.geometry import Polygon, mapping

TEST_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TEST_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services import zones  # noqa: E402


def _write_ndvi_raster(path: Path, *, data: np.ndarray | None = None) -> None:
    if data is None:
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


def _write_zone_raster(path: Path, data: np.ndarray) -> None:
    transform = from_origin(0, 30, zones.DEFAULT_SCALE, zones.DEFAULT_SCALE)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="uint8",
        crs=zones.DEFAULT_EXPORT_CRS,
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(data, 1)


def test_compute_ndvi_preserves_source_mask(monkeypatch) -> None:
    class FakeNdviBand:
        def __init__(self):
            self.renamed: str | None = None
            self.update_mask_arg: object | None = None

        def rename(self, name: str):
            self.renamed = name
            return self

        def updateMask(self, value: object):
            self.update_mask_arg = value
            return self

    class FakeMask:
        def __init__(self, label: str):
            self.label = label
            self.and_args: list["FakeMask"] = []

        def And(self, other: "FakeMask"):
            self.and_args.append(other)
            return self

    class FakeAddResult:
        def __init__(self, parent: "FakeBands"):
            self.parent = parent

        def add(self, value: float):
            self.parent.add_constants.append(value)
            return self

    class FakeSubtractResult:
        def __init__(self, parent: "FakeBands"):
            self.parent = parent

        def divide(self, denominator: object):
            self.parent.divide_args.append(denominator)
            return self.parent.ndvi_band

    class FakeBands:
        def __init__(self, label: str, mask_obj: FakeMask, ndvi_band: FakeNdviBand):
            self.label = label
            self.mask_obj = mask_obj
            self.ndvi_band = ndvi_band
            self.to_float_calls = 0
            self.subtract_args: list[str] = []
            self.add_args: list[str] = []
            self.add_constants: list[float] = []
            self.divide_args: list[object] = []

        def toFloat(self):
            self.to_float_calls += 1
            return self

        def subtract(self, other: "FakeBands"):
            self.subtract_args.append(other.label)
            return FakeSubtractResult(self)

        def add(self, other: "FakeBands"):
            self.add_args.append(other.label)
            return FakeAddResult(self)

        def mask(self):
            return self.mask_obj

    class FakeImage:
        def __init__(self):
            self.mask_obj = FakeMask("image")
            self.ndvi_band = FakeNdviBand()
            self.selected: list[str] = []
            self.bands: dict[str, FakeBands] = {}
            self.mask_calls = 0
            self.band_masks: dict[str, FakeMask] = {}

        def select(self, band: str):
            self.selected.append(band)
            band_mask = FakeMask(band)
            self.band_masks[band] = band_mask
            band_obj = FakeBands(band, band_mask, self.ndvi_band)
            self.bands[band] = band_obj
            return band_obj

        def mask(self):
            self.mask_calls += 1
            return self.mask_obj

    fake_image = FakeImage()

    result = zones.compute_ndvi(fake_image)

    assert fake_image.selected == ["B8", "B4"]
    assert fake_image.bands["B8"].to_float_calls == 1
    assert fake_image.bands["B4"].to_float_calls == 1
    assert fake_image.bands["B8"].subtract_args == ["B4"]
    assert fake_image.bands["B8"].add_args == ["B4"]
    assert fake_image.bands["B8"].add_constants == [1e-6]
    assert fake_image.bands["B8"].divide_args
    assert fake_image.mask_calls == 0
    assert fake_image.ndvi_band.renamed == "NDVI"
    assert fake_image.ndvi_band.update_mask_arg is fake_image.bands["B8"].mask_obj
    assert fake_image.bands["B8"].mask_obj.and_args == [fake_image.bands["B4"].mask_obj]
    assert result is fake_image.ndvi_band


def test_download_image_to_path_handles_zipped_payload(
    monkeypatch, tmp_path: Path
) -> None:
    source = tmp_path / "source_ndvi.tif"
    _write_ndvi_raster(source)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.write(source, arcname="mean_ndvi.tif")
    payload = buffer.getvalue()

    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()

        @property
        def headers(self):
            return {"Content-Type": "application/zip"}

    class FakeImage:
        def getDownloadURL(self, params):
            return "https://example.com/download"

    class FakeTask:
        def start(self):
            pass

    monkeypatch.setattr(zones, "urlopen", lambda url: FakeResponse(payload))
    monkeypatch.setattr(zones, "_geometry_region", lambda geometry: [[0, 0]])
    monkeypatch.setattr(
        zones.ee, "Geometry", SimpleNamespace(Polygon=lambda coords: coords)
    )
    monkeypatch.setattr(
        zones.ee.batch.Export.image,
        "toDrive",
        lambda **_kwargs: FakeTask(),
    )

    target = tmp_path / "downloaded_ndvi.tif"
    result = zones._download_image_to_path(FakeImage(), object(), target)

    assert result.path == target
    assert target.exists()
    with rasterio.open(target) as dataset:
        assert dataset.count == 1


def test_download_image_to_path_merges_multipolygon_region(
    monkeypatch, tmp_path: Path
) -> None:
    polygon_a_coords = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 0.0],
    ]
    polygon_b_coords = [
        [1.0, 0.0],
        [1.0, 1.0],
        [2.0, 1.0],
        [2.0, 0.0],
        [1.0, 0.0],
    ]
    multipolygon_info = {
        "type": "MultiPolygon",
        "coordinates": [
            [polygon_a_coords],
            [polygon_b_coords],
        ],
    }

    polygon_a = Polygon(polygon_a_coords)
    polygon_b = Polygon(polygon_b_coords)
    merged_polygon = polygon_a.union(polygon_b)

    def _lists(value):
        if isinstance(value, tuple):
            return [_lists(item) for item in value]
        if isinstance(value, list):
            return [_lists(item) for item in value]
        return value

    merged_info = {
        "type": "Polygon",
        "coordinates": _lists(mapping(merged_polygon)["coordinates"]),
    }

    dissolve_called = {"value": False}

    class FakeGeometry:
        def __init__(self, info, dissolved=None):
            self._info = info
            self._dissolved = dissolved

        def getInfo(self):
            return self._info

        def dissolve(self):
            dissolve_called["value"] = True
            if self._dissolved is None:
                return self
            return FakeGeometry(self._dissolved)

    geometry = FakeGeometry(multipolygon_info, dissolved=merged_info)

    captured = {}

    class FakeImage:
        def getDownloadURL(self, params):
            captured.update(params)
            return "https://example.com/download"

    class FakeResponse(io.BytesIO):
        def __init__(self):
            super().__init__(b"fake-tiff")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()

        @property
        def headers(self):
            return {"Content-Type": "image/tiff"}

    class FakeTask:
        def start(self):
            pass

    monkeypatch.setattr(zones, "urlopen", lambda url: FakeResponse())
    monkeypatch.setattr(
        zones.ee, "Geometry", SimpleNamespace(Polygon=lambda coords: coords)
    )
    monkeypatch.setattr(
        zones.ee.batch.Export.image,
        "toDrive",
        lambda **_kwargs: FakeTask(),
    )

    target = tmp_path / "merged_region.tif"
    result = zones._download_image_to_path(FakeImage(), geometry, target)

    assert result.path == target
    assert target.exists()
    assert dissolve_called["value"] is True
    assert "region" in captured

    region_coords = captured["region"]
    holes = region_coords[1:] if len(region_coords) > 1 else None
    result_polygon = Polygon(region_coords[0], holes=holes)

    assert result_polygon.equals(merged_polygon)
    assert result_polygon.contains(polygon_a.centroid)
    assert result_polygon.contains(polygon_b.centroid)


def test_classify_local_zones_generates_outputs(tmp_path: Path) -> None:
    ndvi_path = tmp_path / "mean_ndvi.tif"
    data = np.linspace(0.1, 0.9, 25, dtype=np.float32).reshape(5, 5)
    data[-1, -1] = np.nan
    _write_ndvi_raster(ndvi_path, data=data)

    artifacts, metadata = zones._classify_local_zones(
        ndvi_path,
        working_dir=tmp_path,
        n_classes=5,
        min_mapping_unit_ha=0.0,
        smooth_radius_m=0,
        open_radius_m=0,
        close_radius_m=0,
        include_stats=True,
    )

    assert Path(artifacts.raster_path).exists()
    assert Path(artifacts.mean_ndvi_path).exists()
    assert Path(artifacts.vector_path).exists()
    assert artifacts.vector_components
    assert artifacts.zonal_stats_path is not None
    assert Path(artifacts.zonal_stats_path).exists()

    with rasterio.open(artifacts.raster_path) as classified:
        classes = np.unique(classified.read(1))
    assert set(classes) >= {0, 1, 2, 3, 4, 5}

    assert "percentile_thresholds" in metadata
    assert len(metadata["percentile_thresholds"]) == 4
    assert all(np.isfinite(metadata["percentile_thresholds"]))
    assert metadata["zones"]

    with open(artifacts.zonal_stats_path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
    assert any(float(row["mean_ndvi"]) > 0 for row in rows)


def test_prepare_selected_period_artifacts_percentiles(
    monkeypatch, tmp_path: Path
) -> None:
    class FakeStatImage:
        def __init__(self):
            self.renamed: list[str] = []

        def rename(self, name: str):
            self.renamed.append(name)
            return self

    def fake_composites(_geometry, months, *_args, **_kwargs):
        return [("2024-01", object())], [], {"composite_mode": "monthly"}

    fake_stats = {
        "mean": FakeStatImage(),
        "median": FakeStatImage(),
        "std": FakeStatImage(),
        "cv": FakeStatImage(),
    }

    called = {"count": 0}

    def fake_classify(ndvi_path, **_kwargs):
        called["count"] += 1
        raster = tmp_path / "zones_classified.tif"
        vector = tmp_path / "zones.shp"
        stats = tmp_path / "zones.csv"
        _write_ndvi_raster(raster)
        vector.write_text("")
        stats.write_text("")
        artifacts = zones.ZoneArtifacts(
            raster_path=str(raster),
            mean_ndvi_path=str(ndvi_path),
            vector_path=str(vector),
            vector_components={"shp": str(vector)},
            zonal_stats_path=str(stats),
            working_dir=str(tmp_path),
        )
        return artifacts, {"classification_method": "percentiles"}

    monkeypatch.setattr(zones, "_build_composite_series", fake_composites)
    monkeypatch.setattr(zones, "_compute_ndvi", lambda image: image)
    monkeypatch.setattr(zones, "_ndvi_temporal_stats", lambda images: fake_stats)
    monkeypatch.setattr(zones, "_stability_mask", lambda *args, **kwargs: "stability")
    monkeypatch.setattr(zones, "_classify_local_zones", fake_classify)

    def _fail_multiindex(*_args, **_kwargs):
        raise RuntimeError("multiindex should not run")

    monkeypatch.setattr(zones, "_build_multiindex_zones", _fail_multiindex)

    def fake_download(image, _geometry, target):
        if target.name == "mean_ndvi.tif":
            _write_ndvi_raster(target)
        return zones.ImageExportResult(path=target, task=None)

    monkeypatch.setattr(zones, "_download_image_to_path", fake_download)

    aoi = {
        "type": "Polygon",
        "coordinates": [[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]],
    }  # noqa: E501

    artifacts, metadata = zones._prepare_selected_period_artifacts(
        aoi,
        geometry=aoi,
        working_dir=tmp_path,
        months=["2024-01"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
        cloud_prob_max=40,
        n_classes=5,
        cv_mask_threshold=zones.DEFAULT_CV_THRESHOLD,
        apply_stability_mask=True,
        min_mapping_unit_ha=1.5,
        smooth_radius_m=0,
        open_radius_m=0,
        close_radius_m=0,
        simplify_tol_m=0,
        simplify_buffer_m=0,
        method="ndvi_percentiles",
        sample_size=zones.DEFAULT_SAMPLE_SIZE,
        include_stats=True,
    )

    assert called["count"] == 1
    assert metadata["zone_method"] == "ndvi_percentiles"
    assert metadata["classification_method"] == "percentiles"
    assert metadata["stability_mask_applied"] is True
    assert Path(artifacts.raster_path).exists()
    assert Path(artifacts.mean_ndvi_path).exists()
    assert metadata["downloaded_mean_ndvi"] == artifacts.mean_ndvi_path
    assert metadata["mean_ndvi_export_task"] == {}


def test_prepare_selected_period_artifacts_percentiles_without_fallback(
    monkeypatch, tmp_path: Path
) -> None:
    class FakeReducerResult:
        def __init__(self, payload: dict[str, float]):
            self._payload = dict(payload)

        def getInfo(self):
            return dict(self._payload)

    class FakeMask:
        def reduceRegion(self, *args, **kwargs):
            return FakeReducerResult({"NDVI": 42})

        def reduce(self, *_args, **_kwargs):  # pragma: no cover - interface guard
            return FakeReducerResult({"NDVI": 42})

        def Not(self):  # pragma: no cover - interface guard
            return self

        def connectedPixelCount(self, *args, **kwargs):  # pragma: no cover
            return self

        def gt(self, *_args, **_kwargs):  # pragma: no cover
            return self

        def selfMask(self):  # pragma: no cover - interface guard
            return self

    class FakeMeanImage:
        def __init__(self):
            self.band_names = ["NDVI_mean"]
            self.last_mask = None
            self.selected: list[int] = []

        def toFloat(self):
            return self

        def updateMask(self, mask):
            self.last_mask = mask
            return self

        def select(self, selectors):
            self.selected = list(selectors)
            return self

        def rename(self, name):
            if isinstance(name, (list, tuple)):
                self.band_names = list(name)
            else:
                self.band_names = [name]
            return self

        def bandNames(self):  # pragma: no cover - helpers for assertions
            return SimpleNamespace(
                get=lambda idx: self.band_names[idx],
                getInfo=lambda: list(self.band_names),
                size=lambda: len(self.band_names),
            )

        def clip(self, _geometry):  # pragma: no cover - interface guard
            return self

        def mask(self):  # pragma: no cover - interface guard
            return FakeMask()

        def reduceRegion(self, *args, **kwargs):  # pragma: no cover
            return FakeReducerResult({"NDVI_min": 0.1, "NDVI_max": 0.9})

        def unmask(self, _fill):  # pragma: no cover - interface guard
            return self

        def selfMask(self):  # pragma: no cover - interface guard
            return self

    def fake_composites(_geometry, months, *_args, **_kwargs):
        return [(months[0], object())], [], {"composite_mode": "monthly"}

    mean_image = FakeMeanImage()
    fake_stats = {
        "mean": mean_image,
        "median": FakeMeanImage(),
        "std": FakeMeanImage(),
        "cv": FakeMeanImage(),
    }

    def fake_ndvi_stats(_images):
        return fake_stats

    def fail_image_collection(*_args, **_kwargs):
        raise RuntimeError("EE ImageCollection unavailable in test")

    def fake_download(image, _geometry, target):
        _write_ndvi_raster(target)
        return zones.ImageExportResult(path=target, task=None)

    def fake_breaks(image, _geometry, n_classes):
        bands = []
        if hasattr(image, "band_names"):
            bands = list(getattr(image, "band_names"))
        elif hasattr(image, "bandNames"):
            info = getattr(image.bandNames(), "getInfo", lambda: [])()
            bands = list(info) if info is not None else []
        if "NDVI" not in bands:
            return []
        return [0.2 * (idx + 1) for idx in range(max(n_classes - 1, 0))]

    class FakeEeList(list):
        def getInfo(self):
            return list(self)

    monkeypatch.setattr(zones, "_build_composite_series", fake_composites)
    monkeypatch.setattr(zones, "_compute_ndvi", lambda image: FakeMeanImage())
    monkeypatch.setattr(zones, "_ndvi_temporal_stats", fake_ndvi_stats)
    monkeypatch.setattr(zones, "_stability_mask", lambda *_args, **_kwargs: "stability")
    monkeypatch.setattr(zones, "_download_image_to_path", fake_download)
    monkeypatch.setattr(zones, "robust_quantile_breaks", fake_breaks)
    monkeypatch.setattr(zones, "_to_ee_geometry", lambda geom: geom)
    monkeypatch.setattr(zones.ee, "ImageCollection", fail_image_collection)
    monkeypatch.setattr(zones.ee, "List", lambda values=None: FakeEeList(values or []))

    aoi = {
        "type": "Polygon",
        "coordinates": [[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]],
    }

    artifacts, metadata = zones._prepare_selected_period_artifacts(
        aoi,
        geometry=aoi,
        working_dir=tmp_path,
        months=["2024-01"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
        cloud_prob_max=40,
        n_classes=3,
        cv_mask_threshold=zones.DEFAULT_CV_THRESHOLD,
        apply_stability_mask=True,
        min_mapping_unit_ha=0.0,
        smooth_radius_m=0,
        open_radius_m=0,
        close_radius_m=0,
        simplify_tol_m=0,
        simplify_buffer_m=0,
        method="ndvi_percentiles",
        sample_size=zones.DEFAULT_SAMPLE_SIZE,
        include_stats=False,
    )

    assert Path(artifacts.raster_path).exists()
    assert metadata["zone_method"] == "ndvi_percentiles"
    assert metadata["classification_method"] == "percentiles"
    assert metadata["kmeans_fallback_applied"] is False
    assert metadata["percentile_thresholds"] == [0.2, 0.4]


def test_prepare_selected_period_artifacts_ndvi_kmeans(
    monkeypatch, tmp_path: Path
) -> None:
    class FakeStatImage:
        def __init__(self):
            self.renamed: list[str] = []

        def rename(self, name: str):
            self.renamed.append(name)
            return self

    def fake_composites(_geometry, months, *_args, **_kwargs):
        return [("2024-01", object())], [], {"composite_mode": "monthly"}

    fake_stats = {
        "mean": FakeStatImage(),
        "median": FakeStatImage(),
        "std": FakeStatImage(),
        "cv": FakeStatImage(),
    }

    class FakeZoneImage:
        def rename(self, _name: str):
            return self

    ndvi_kmeans_calls = {"count": 0}

    def fake_kmeans_classify(*_args, **_kwargs):
        ndvi_kmeans_calls["count"] += 1
        return FakeZoneImage()

    def fake_cleanup(image, *_args, **_kwargs):
        return zones.CleanupResult(
            image=image,
            applied_operations={
                "smooth": False,
                "open": False,
                "close": False,
                "min_mapping_unit": True,
            },
            executed_operations={
                "smooth": False,
                "open": False,
                "close": False,
                "min_mapping_unit": True,
            },
            fallback_applied=False,
            fallback_removed=[],
        )

    monkeypatch.setattr(zones, "_build_composite_series", fake_composites)
    monkeypatch.setattr(zones, "_compute_ndvi", lambda image: image)
    monkeypatch.setattr(zones, "_ndvi_temporal_stats", lambda images: fake_stats)
    monkeypatch.setattr(zones, "_stability_mask", lambda *args, **kwargs: "stability")

    def _fail_percentiles(*_args, **_kwargs):
        raise RuntimeError("percentile path should not run")

    monkeypatch.setattr(zones, "_classify_local_zones", _fail_percentiles)

    def _fail_multiindex(*_args, **_kwargs):
        raise RuntimeError("multiindex path should not run")

    monkeypatch.setattr(zones, "_build_multiindex_zones", _fail_multiindex)
    monkeypatch.setattr(
        zones, "_build_multiindex_zones_with_features", _fail_multiindex
    )
    monkeypatch.setattr(zones, "kmeans_classify", fake_kmeans_classify)
    monkeypatch.setattr(zones, "_apply_cleanup_with_fallback_tracking", fake_cleanup)

    def fake_download(image, _geometry, target):
        if target.name == "mean_ndvi.tif":
            _write_ndvi_raster(target)
        elif target.name == "zones_classified.tif":
            data = np.array(
                [
                    [1, 1, 2],
                    [2, 3, 3],
                    [0, 0, 3],
                ],
                dtype=np.uint8,
            )
            _write_zone_raster(target, data)
        else:  # pragma: no cover - defensive
            raise AssertionError(target.name)
        return zones.ImageExportResult(path=target, task=None)

    monkeypatch.setattr(zones, "_download_image_to_path", fake_download)

    aoi = {
        "type": "Polygon",
        "coordinates": [[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]],
    }  # noqa: E501

    artifacts, metadata = zones._prepare_selected_period_artifacts(
        aoi,
        geometry=aoi,
        working_dir=tmp_path,
        months=["2024-01"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
        cloud_prob_max=40,
        n_classes=4,
        cv_mask_threshold=zones.DEFAULT_CV_THRESHOLD,
        apply_stability_mask=True,
        min_mapping_unit_ha=1.5,
        smooth_radius_m=0,
        open_radius_m=0,
        close_radius_m=0,
        simplify_tol_m=0,
        simplify_buffer_m=0,
        method="ndvi_kmeans",
        sample_size=4321,
        include_stats=True,
    )

    assert ndvi_kmeans_calls["count"] == 1
    assert metadata["zone_method"] == "ndvi_kmeans"
    assert metadata["classification_method"] == "ndvi_kmeans"
    assert metadata["percentile_thresholds"] == []
    assert metadata["mean_ndvi_export_task"] == {}
    assert metadata["zone_raster_export_task"] == {}
    assert metadata["kmeans_sample_size"] == 4321
    assert metadata["kmeans_features"] == {
        "method": "kmeans",
        "features": ["NDVI"],
    }
    assert metadata["kmeans_feature_count"] == 1
    assert metadata["stability_mask_applied"] is True
    assert Path(artifacts.raster_path).exists()
    assert Path(metadata["downloaded_zone_raster"]).exists()
    assert Path(artifacts.mean_ndvi_path).exists()
    assert metadata["downloaded_mean_ndvi"] == artifacts.mean_ndvi_path


def test_prepare_selected_period_artifacts_multiindex(
    monkeypatch, tmp_path: Path
) -> None:
    class FakeStatImage:
        def __init__(self):
            self.renamed: list[str] = []

        def rename(self, name: str):
            self.renamed.append(name)
            return self

    def fake_composites(_geometry, months, *_args, **_kwargs):
        return [("2024-01", object())], [], {"composite_mode": "monthly"}

    fake_stats = {
        "mean": FakeStatImage(),
        "median": FakeStatImage(),
        "std": FakeStatImage(),
        "cv": FakeStatImage(),
    }

    class FakeZoneImage:
        def rename(self, _name: str):
            return self

    multiindex_calls = {"count": 0}

    def fake_multiindex(*_args, **_kwargs):
        multiindex_calls["count"] += 1
        image = FakeZoneImage()
        cleanup = zones.CleanupResult(
            image=image,
            applied_operations={
                "smooth": False,
                "open": False,
                "close": False,
                "min_mapping_unit": False,
            },
            executed_operations={
                "smooth": False,
                "open": False,
                "close": False,
                "min_mapping_unit": False,
            },
            fallback_applied=False,
            fallback_removed=[],
        )
        return image, {"NDVI": object(), "NDRE": object()}, cleanup

    monkeypatch.setattr(zones, "_build_composite_series", fake_composites)
    monkeypatch.setattr(zones, "_compute_ndvi", lambda image: image)
    monkeypatch.setattr(zones, "_ndvi_temporal_stats", lambda images: fake_stats)
    monkeypatch.setattr(zones, "_stability_mask", lambda *args, **kwargs: "stability")

    def _fail_percentiles(*_args, **_kwargs):
        raise RuntimeError("percentile path should not run")

    monkeypatch.setattr(zones, "_classify_local_zones", _fail_percentiles)
    monkeypatch.setattr(zones, "_build_multiindex_zones", fake_multiindex)

    def _fail_features(*_args, **_kwargs):
        raise RuntimeError("features not supplied")

    monkeypatch.setattr(zones, "_build_multiindex_zones_with_features", _fail_features)

    def fake_download(image, _geometry, target):
        if target.name == "mean_ndvi.tif":
            _write_ndvi_raster(target)
        elif target.name == "zones_classified.tif":
            data = np.array(
                [
                    [1, 1, 2],
                    [2, 3, 3],
                    [0, 0, 3],
                ],
                dtype=np.uint8,
            )
            _write_zone_raster(target, data)
        else:  # pragma: no cover - defensive
            raise AssertionError(target.name)
        return zones.ImageExportResult(path=target, task=None)

    monkeypatch.setattr(zones, "_download_image_to_path", fake_download)

    aoi = {
        "type": "Polygon",
        "coordinates": [[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]],
    }  # noqa: E501

    artifacts, metadata = zones._prepare_selected_period_artifacts(
        aoi,
        geometry=aoi,
        working_dir=tmp_path,
        months=["2024-01"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
        cloud_prob_max=40,
        n_classes=4,
        cv_mask_threshold=zones.DEFAULT_CV_THRESHOLD,
        apply_stability_mask=True,
        min_mapping_unit_ha=1.5,
        smooth_radius_m=0,
        open_radius_m=0,
        close_radius_m=0,
        simplify_tol_m=0,
        simplify_buffer_m=0,
        method="multiindex_kmeans",
        sample_size=1234,
        include_stats=True,
    )

    assert multiindex_calls["count"] == 1
    assert metadata["zone_method"] == "multiindex_kmeans"
    assert metadata["classification_method"] == "multiindex_kmeans"
    assert metadata["percentile_thresholds"] == []
    assert metadata["multiindex_feature_names"] == ["NDRE", "NDVI"]
    assert metadata["multiindex_sample_size"] == 1234
    assert Path(artifacts.raster_path).exists()
    assert Path(metadata["downloaded_zone_raster"]).exists()
    assert Path(artifacts.mean_ndvi_path).exists()
    assert metadata["downloaded_mean_ndvi"] == artifacts.mean_ndvi_path
    assert metadata["mean_ndvi_export_task"] == {}
    assert metadata["zone_raster_export_task"] == {}


def test_cleanup_helper_rolls_back_min_mapping_unit(monkeypatch) -> None:
    class FakeReducerResult:
        def __init__(self, histogram: dict[int, int]):
            self._histogram = histogram

        def getInfo(self):
            return {"zone": {str(key): value for key, value in self._histogram.items()}}

    class FakeCleanupImage:
        def __init__(self, histogram: dict[int, int]):
            self._histogram = dict(histogram)

        def reduceRegion(self, **_kwargs):
            return FakeReducerResult(self._histogram)

        def rename(self, _name: str):  # pragma: no cover - interface parity
            return self

        def clip(self, _geometry):  # pragma: no cover - interface parity
            return self

    def fake_apply_cleanup(_classified, _geometry, **kwargs):
        min_mapping_unit = kwargs.get("min_mapping_unit_ha", 0)
        if min_mapping_unit and min_mapping_unit > 0:
            histogram = {1: 8, 2: 6}
        else:
            histogram = {1: 8, 2: 6, 3: 2}
        return FakeCleanupImage(histogram)

    fake_ee = SimpleNamespace(Reducer=SimpleNamespace(frequencyHistogram=lambda: None))
    monkeypatch.setattr(zones, "ee", fake_ee)
    monkeypatch.setattr(zones, "_apply_cleanup", fake_apply_cleanup)

    base_image = FakeCleanupImage({1: 8, 2: 6, 3: 2})
    result = zones._apply_cleanup_with_fallback_tracking(
        base_image,
        geometry=object(),
        n_classes=3,
        smooth_radius_m=0,
        open_radius_m=0,
        close_radius_m=0,
        min_mapping_unit_ha=1.5,
    )

    assert result.fallback_applied is True
    assert result.fallback_removed == ["min_mapping_unit"]
    assert result.applied_operations["min_mapping_unit"] is False
    assert result.executed_operations["min_mapping_unit"] is True

    histogram = result.image.reduceRegion().getInfo()["zone"]
    populated = [key for key in histogram if int(key) > 0]
    assert len(populated) == 3


def test_classify_local_zones_quantized_values_expand_to_requested_classes(
    tmp_path: Path,
) -> None:
    ndvi_path = tmp_path / "mean_ndvi_two_values.tif"
    data = np.full((6, 6), 0.1, dtype=np.float32)
    data[3:, :] = 0.7
    _write_ndvi_raster(ndvi_path, data=data)

    artifacts, metadata = zones._classify_local_zones(
        ndvi_path,
        working_dir=tmp_path,
        n_classes=5,
        min_mapping_unit_ha=0.0,
        smooth_radius_m=0,
        open_radius_m=0,
        close_radius_m=0,
        include_stats=False,
    )

    with rasterio.open(artifacts.raster_path) as classified:
        classes = np.unique(classified.read(1))

    populated_classes = {int(value) for value in classes if value > 0}
    assert populated_classes == {1, 2, 3, 4, 5}

    assert metadata["requested_zone_count"] == 5
    assert metadata["effective_zone_count"] == 5
    assert metadata["final_zone_count"] == 5
    assert metadata["classification_method"] == "kmeans"
    assert metadata["kmeans_fallback_applied"] is True


def test_classify_local_zones_recovers_sparse_percentiles(tmp_path: Path) -> None:
    ndvi_path = tmp_path / "mean_ndvi_sparse_percentiles.tif"
    data = np.full((20, 20), 0.1, dtype=np.float32)
    data[:2, :2] = 0.2
    data[0, 0] = 0.9
    _write_ndvi_raster(ndvi_path, data=data)

    artifacts, metadata = zones._classify_local_zones(
        ndvi_path,
        working_dir=tmp_path,
        n_classes=3,
        min_mapping_unit_ha=0.0,
        smooth_radius_m=0,
        open_radius_m=0,
        close_radius_m=0,
        include_stats=False,
    )

    with rasterio.open(artifacts.raster_path) as classified:
        classes = np.unique(classified.read(1))

    populated_classes = {int(value) for value in classes if value > 0}
    assert populated_classes == {1, 2, 3}
    assert metadata["effective_zone_count"] == 3
    assert metadata["final_zone_count"] == 3

    thresholds = metadata["percentile_thresholds"]
    assert np.isclose(thresholds[0], 0.15, atol=1e-6)
    assert np.isclose(thresholds[1], 0.55, atol=1e-6)


def test_classify_local_zones_includes_minimum_in_first_class(tmp_path: Path) -> None:
    ndvi_path = tmp_path / "mean_ndvi_equal_min.tif"
    custom_data = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2],
        ],
        dtype=np.float32,
    )
    _write_ndvi_raster(ndvi_path, data=custom_data)

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

    with rasterio.open(artifacts.raster_path) as classified:
        classes = np.unique(classified.read(1))

    assert set(classes) >= {1, 2, 3, 4}
    assert np.isclose(metadata["percentile_thresholds"][0], float(custom_data.min()))


def test_classify_local_zones_preserves_classes_without_smoothing(
    tmp_path: Path,
) -> None:
    ndvi_path = tmp_path / "mean_ndvi_minimal_smoothing.tif"
    values = np.linspace(0.05, 0.95, 36, dtype=np.float32).reshape(6, 6)
    _write_ndvi_raster(ndvi_path, data=values)

    artifacts, metadata = zones._classify_local_zones(
        ndvi_path,
        working_dir=tmp_path,
        n_classes=4,
        min_mapping_unit_ha=0.0,
        smooth_radius_m=0.0,
        open_radius_m=0.0,
        close_radius_m=0.0,
        include_stats=False,
    )

    with rasterio.open(artifacts.raster_path) as classified:
        classes = np.unique(classified.read(1))

    populated_classes = {int(value) for value in classes if value > 0}
    assert populated_classes == {1, 2, 3, 4}

    smoothing_info = metadata.get("smoothing_parameters", {})
    assert smoothing_info.get("fallback_applied") is False
    assert smoothing_info.get("skipped_steps") == []
    assert smoothing_info.get("applied", {}).get("smooth_radius_m") == 0.0
    assert smoothing_info.get("applied", {}).get("open_radius_m") == 0.0
    assert smoothing_info.get("applied", {}).get("close_radius_m") == 0.0
    assert smoothing_info.get("applied", {}).get("min_mapping_unit_ha") == 0.0
    executed_info = smoothing_info.get("executed", {})
    assert executed_info.get("smooth_radius_m") is False
    assert executed_info.get("open_radius_m") is False
    assert executed_info.get("close_radius_m") is False
    assert executed_info.get("min_mapping_unit_ha") is False

    assert metadata["requested_zone_count"] == 4
    assert metadata["effective_zone_count"] == 4
    assert metadata["final_zone_count"] == 4


def test_classify_local_zones_preserves_nodata_border(tmp_path: Path) -> None:
    ndvi_path = tmp_path / "mean_ndvi_masked_border.tif"
    data = np.full((6, 6), 0.3, dtype=np.float32)
    data[0, :] = -9999.0
    data[-1, :] = -9999.0
    data[:, 0] = -9999.0
    data[:, -1] = -9999.0
    data[2:4, 2:4] = np.array([[0.6, 0.7], [0.8, 0.9]], dtype=np.float32)
    _write_ndvi_raster(ndvi_path, data=data)

    artifacts, metadata = zones._classify_local_zones(
        ndvi_path,
        working_dir=tmp_path,
        n_classes=3,
        min_mapping_unit_ha=0.0,
        smooth_radius_m=20.0,
        open_radius_m=0.0,
        close_radius_m=0.0,
        include_stats=False,
    )

    with rasterio.open(artifacts.raster_path) as classified:
        classified_data = classified.read(1)

    assert np.all(classified_data[0, :] == 0)
    assert np.all(classified_data[-1, :] == 0)
    assert np.all(classified_data[:, 0] == 0)
    assert np.all(classified_data[:, -1] == 0)

    reader = shapefile.Reader(str(artifacts.vector_path))
    shapes = list(reader.iterShapeRecords())
    assert shapes, "Expected at least one zone geometry"
    for shape_record in shapes:
        minx, miny, maxx, maxy = shape_record.shape.bbox
        assert minx >= 10.0 - 1e-6
        assert maxx <= 50.0 + 1e-6
        assert maxy <= 20.0 + 1e-6
        assert miny >= -20.0 - 1e-6

    assert metadata["final_zone_count"] == 3


def test_classify_local_zones_relaxes_mmu_when_classes_removed(tmp_path: Path) -> None:
    ndvi_path = tmp_path / "mean_ndvi_sparse_classes.tif"
    data = np.full((30, 30), 0.1, dtype=np.float32)
    data[15:, :] = 0.5
    data[5:10, 5:10] = 0.2
    data[10:12, 20:22] = 0.3
    data[2:4, 25:27] = 0.4
    _write_ndvi_raster(ndvi_path, data=data)

    artifacts, metadata = zones._classify_local_zones(
        ndvi_path,
        working_dir=tmp_path,
        n_classes=5,
        min_mapping_unit_ha=1.5,
        smooth_radius_m=0,
        open_radius_m=0,
        close_radius_m=0,
        include_stats=False,
    )

    with rasterio.open(artifacts.raster_path) as classified:
        raster_classes = np.unique(classified.read(1))

    non_zero_classes = {int(value) for value in raster_classes if value > 0}
    assert non_zero_classes == {1, 2, 3, 4, 5}

    smoothing_info = metadata.get("smoothing_parameters", {})
    assert smoothing_info.get("fallback_applied") is True
    assert "min_mapping_unit_ha" in smoothing_info.get("skipped_steps", [])
    assert smoothing_info.get("applied", {}).get("min_mapping_unit_ha") == 0.0
    assert metadata["min_mapping_unit_applied"] is False


def test_classify_local_zones_uses_kmeans_fallback_for_sparse_bins(
    tmp_path: Path, monkeypatch
) -> None:
    ndvi_path = tmp_path / "mean_ndvi_bimodal.tif"
    data = np.full((10, 10), 0.1, dtype=np.float32)
    data[0, :5] = np.array([0.1, 0.11, 0.12, 0.13, 0.14], dtype=np.float32)
    data[5:, :] = 0.9
    data[-1, -5:] = np.array([0.86, 0.88, 0.9, 0.92, 0.94], dtype=np.float32)

    original_unique = zones.np.unique
    trigger_state = {"count": 0}

    def _forced_unique(values, *args, **kwargs):
        result = original_unique(values, *args, **kwargs)
        if (
            isinstance(values, np.ndarray)
            and values.dtype == np.int16
            and values.ndim == 1
            and result.size >= 2
        ):
            trigger_state["count"] += 1
            if trigger_state["count"] >= 2:
                return np.array([1, 5], dtype=result.dtype)
        return result

    monkeypatch.setattr(zones.np, "unique", _forced_unique)
    _write_ndvi_raster(ndvi_path, data=data)

    artifacts, metadata = zones._classify_local_zones(
        ndvi_path,
        working_dir=tmp_path,
        n_classes=5,
        min_mapping_unit_ha=0.0,
        smooth_radius_m=0,
        open_radius_m=0,
        close_radius_m=0,
        include_stats=False,
    )

    assert trigger_state["count"] >= 2

    with rasterio.open(artifacts.raster_path) as classified:
        classified_data = classified.read(1)

    populated_classes = sorted(
        {int(value) for value in np.unique(classified_data) if value > 0}
    )
    assert populated_classes == [1, 2, 3, 4, 5]

    assert metadata["kmeans_fallback_applied"] is True
    assert metadata["classification_method"] == "kmeans"
    assert metadata["percentile_thresholds"] == []
    assert len(metadata["kmeans_cluster_centers"]) == 5
    assert metadata["kmeans_cluster_centers"] == sorted(
        metadata["kmeans_cluster_centers"]
    )


def test_export_selected_period_zones_returns_local_paths(
    monkeypatch, tmp_path: Path
) -> None:
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
    assert Path(paths["mean_ndvi"]).exists()
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
    zones_in_file = {record.record[zone_index] for record in reader.iterShapeRecords()}
    assert set(zones_in_file) <= set(range(1, 6))


def test_export_selected_period_zones_preserves_thresholds_for_kmeans(
    monkeypatch, tmp_path: Path
) -> None:
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
        metadata.update(
            {
                "used_months": list(kwargs["months"]),
                "skipped_months": [],
                "zone_method": kwargs["method"],
                "percentile_thresholds": [],
            }
        )
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
        method="ndvi_kmeans",
    )

    assert result["metadata"]["zone_method"] == "ndvi_kmeans"
    assert "palette" in result and result["palette"]
    assert "thresholds" in result
    assert result["thresholds"] == []


def test_classify_by_percentiles_handles_duplicate_thresholds(monkeypatch) -> None:
    reducer_payload = {
        "cut_01": 0.2,
        "cut_02": 0.2,
        "cut_03": 0.5,
        "cut_04": 0.8,
    }

    class FakeReducerResult:
        def __init__(self, data: dict[str, float]):
            self._data = data

        def getInfo(self):
            return dict(self._data)

    class FakeEEString(str):
        def getInfo(self):
            return str(self)

    class FakeEEList(list):
        def __init__(self, values):
            super().__init__(values)

        def get(self, index):
            return self[index]

        def iterate(self, func, first):
            result = first
            for value in self:
                result = func(result, value)
            return result

    class FakeNumber(float):
        pass

    class FakeImage:
        def __init__(self, values, reducer_result, band_name: str = "ndvi"):
            self.values = list(values)
            self.reducer_result = reducer_result
            self.band_name = band_name

        def bandNames(self):
            return FakeEEList([self.band_name])

        def rename(self, name):
            if hasattr(name, "getInfo"):
                self.band_name = name.getInfo()
            else:
                self.band_name = str(name)
            return self

        def reduceRegion(self, **_kwargs):
            return self.reducer_result

        def multiply(self, scalar):
            factor = float(scalar)
            return FakeImage(
                [value * factor for value in self.values],
                self.reducer_result,
                self.band_name,
            )

        def gt(self, threshold):
            thresh = float(threshold)
            return FakeImage(
                [1 if value > thresh else 0 for value in self.values],
                None,
                self.band_name,
            )

        def add(self, other):
            if isinstance(other, FakeImage):
                values = [a + b for a, b in zip(self.values, other.values)]
            else:
                values = [value + float(other) for value in self.values]
            return FakeImage(values, None, self.band_name)

        def toInt(self):
            return FakeImage(
                [int(round(value)) for value in self.values], None, self.band_name
            )

    fake_ee = type(
        "FakeEE",
        (),
        {
            "String": staticmethod(lambda value: FakeEEString(value)),
            "List": staticmethod(lambda values: FakeEEList(values)),
            "Number": staticmethod(lambda value: FakeNumber(value)),
            "Image": staticmethod(lambda value: value),
            "Reducer": type(
                "Reducer",
                (),
                {"percentile": staticmethod(lambda *args, **kwargs: (args, kwargs))},
            )(),
        },
    )

    monkeypatch.setattr(zones, "ee", fake_ee)

    values = [0.1, 0.20000000000000003, 0.3, 0.51, 0.81]
    image = FakeImage(values, FakeReducerResult(reducer_payload))

    classified, thresholds = zones._classify_by_percentiles(
        image, geometry=object(), n_classes=5
    )

    assert all(later > earlier for earlier, later in zip(thresholds, thresholds[1:]))
    assert set(classified.values) == {1, 2, 3, 4, 5}
