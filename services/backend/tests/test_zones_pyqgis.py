"""Tests for PyQGIS zones classification and vectorization."""

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

TEST_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TEST_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def _write_test_ndvi_raster(path: Path, data: np.ndarray | None = None) -> None:
    """Write a test NDVI GeoTIFF with a gradient."""
    if data is None:
        # Create 20x20 raster with gradient
        rows, cols = 20, 20
        data = np.linspace(0.2, 0.8, rows * cols).reshape(rows, cols).astype(np.float32)

    transform = from_origin(0, 20, 1, 1)  # 1m resolution
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": data.dtype,
        "crs": "EPSG:32756",
        "transform": transform,
        "nodata": -9999,
    }

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


@pytest.fixture
def mock_qgis_unavailable(monkeypatch):
    """Mock QGIS as unavailable to test error handling."""
    # Mock the qgis.core module import to raise ImportError
    import sys

    # Store original modules
    original_modules = {}
    qgis_modules = [k for k in sys.modules.keys() if k.startswith("qgis")]
    for mod in qgis_modules:
        original_modules[mod] = sys.modules.pop(mod)

    def restore_modules():
        for mod, val in original_modules.items():
            sys.modules[mod] = val

    # Add finalizer to restore
    monkeypatch.context().addfinalizer(restore_modules)


@pytest.mark.skip(reason="QGIS import mocking causes pytest issues")
def test_pyqgis_import_error_handling(tmp_path, mock_qgis_unavailable):
    """Test that missing PyQGIS raises a helpful error."""
    from app.services.zones_pyqgis import build_zones_with_pyqgis

    ndvi_path = tmp_path / "ndvi.tif"
    _write_test_ndvi_raster(ndvi_path)

    aoi = {
        "type": "Polygon",
        "coordinates": [[[0, 0], [20, 0], [20, 20], [0, 20], [0, 0]]],
    }

    with pytest.raises(RuntimeError, match="PyQGIS is not available"):
        build_zones_with_pyqgis(
            ndvi_tif_path=ndvi_path,
            aoi_geojson=aoi,
            out_dir=tmp_path / "output",
            n_classes=4,
            classifier="kmeans",
            mmu_ha=0.0,
            smooth_radius_m=0.0,
            simplify_tolerance_m=0.0,
            export_format="gpkg",
            seed=42,
        )


@pytest.mark.skip(reason="Complex mocking required - test with actual QGIS")
def test_pyqgis_zones_basic_flow_with_mock(tmp_path, monkeypatch):
    """Test basic PyQGIS zones flow with mocked QGIS (no actual QGIS required)."""
    # This test mocks the QGIS parts to avoid requiring QGIS installation
    from app.services import zones_pyqgis

    ndvi_path = tmp_path / "ndvi.tif"
    _write_test_ndvi_raster(ndvi_path)

    aoi = {
        "type": "Polygon",
        "coordinates": [[[0, 0], [20, 0], [20, 20], [0, 20], [0, 0]]],
    }

    # Mock QGIS imports and processing
    class MockQgsApplication:
        def __init__(self, *args, **kwargs):
            pass

        @staticmethod
        def instance():
            return None

        def initQgis(self):
            pass

        @staticmethod
        def coordinateReferenceSystemRegistry():
            class Registry:
                @staticmethod
                def transformContext():
                    return None

            return Registry()

    class MockProcessing:
        @staticmethod
        def initialize():
            pass

        @staticmethod
        def run(name, params):
            # Create a mock GPKG output
            output_path = Path(params["OUTPUT"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.touch()

    class MockVectorLayer:
        def __init__(self, *args, **kwargs):
            pass

        def isValid(self):
            return True

        def featureCount(self):
            return 10

        def getFeatures(self):
            return []

        def fields(self):
            return []

        def wkbType(self):
            return 3  # Polygon

    class MockVectorFileWriter:
        NoError = 0

        @staticmethod
        def create(*args, **kwargs):
            class Writer:
                def hasError(self):
                    return 0

                def addFeature(self, feature):
                    pass

            return Writer()

        @staticmethod
        def writeAsVectorFormatV3(*args, **kwargs):
            # Create output file
            output_path = Path(args[1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.touch()
            return (0, None)

    # Monkeypatch QGIS modules
    monkeypatch.setattr(
        "app.services.zones_pyqgis.QgsApplication", MockQgsApplication, raising=False
    )
    monkeypatch.setattr(
        "app.services.zones_pyqgis.Processing", MockProcessing, raising=False
    )
    monkeypatch.setattr(
        "app.services.zones_pyqgis.QgsVectorLayer", MockVectorLayer, raising=False
    )
    monkeypatch.setattr(
        "app.services.zones_pyqgis.QgsVectorFileWriter",
        MockVectorFileWriter,
        raising=False,
    )
    monkeypatch.setattr(
        "app.services.zones_pyqgis.processing", MockProcessing, raising=False
    )

    # Bypass the QGIS import check by mocking the imports
    import sys
    from unittest.mock import MagicMock

    mock_qgis_core = MagicMock()
    mock_qgis_pyqt = MagicMock()
    mock_processing = MagicMock()

    sys.modules["qgis"] = MagicMock()
    sys.modules["qgis.core"] = mock_qgis_core
    sys.modules["qgis.PyQt"] = mock_qgis_pyqt
    sys.modules["qgis.PyQt.QtCore"] = MagicMock()
    sys.modules["processing"] = mock_processing
    sys.modules["processing.core"] = MagicMock()
    sys.modules["processing.core.Processing"] = MagicMock()

    # Set up the mocks
    mock_qgis_core.QgsApplication = MockQgsApplication
    mock_qgis_core.QgsVectorLayer = MockVectorLayer
    mock_qgis_core.QgsVectorFileWriter = MockVectorFileWriter
    mock_qgis_core.QgsCoordinateReferenceSystem = MagicMock
    mock_qgis_core.QgsFeature = MagicMock
    mock_qgis_core.QgsField = MagicMock
    mock_qgis_core.QgsFields = MagicMock
    mock_qgis_core.QgsGeometry = MagicMock
    mock_qgis_core.QgsRasterLayer = MagicMock
    mock_qgis_pyqt.QtCore.QVariant = MagicMock
    mock_processing.run = MockProcessing.run

    # Now test the function
    result = zones_pyqgis.build_zones_with_pyqgis(
        ndvi_tif_path=str(ndvi_path),
        aoi_geojson=aoi,
        out_dir=str(tmp_path / "output"),
        n_classes=4,
        classifier="kmeans",
        mmu_ha=0.5,
        smooth_radius_m=10.0,
        simplify_tolerance_m=2.0,
        export_format="gpkg",
        seed=42,
    )

    assert result["ok"] is True
    assert "vector" in result
    assert "metadata" in result
    assert result["metadata"]["classifier"] == "kmeans"
    assert result["metadata"]["n_classes"] == 4
    assert result["metadata"]["mmu_ha"] == 0.5
    assert result["metadata"]["seed"] == 42


# Below tests require actual QGIS installation and are skipped by default
@pytest.mark.skip(reason="PyQGIS tests require QGIS system package installation")
def test_pyqgis_zones_kmeans_classification(tmp_path):
    """Test PyQGIS zones with k-means classification (requires QGIS)."""
    from app.services.zones_pyqgis import build_zones_with_pyqgis

    ndvi_path = tmp_path / "ndvi.tif"
    _write_test_ndvi_raster(ndvi_path)

    aoi = {
        "type": "Polygon",
        "coordinates": [[[0, 0], [20, 0], [20, 20], [0, 20], [0, 0]]],
    }

    result = build_zones_with_pyqgis(
        ndvi_tif_path=ndvi_path,
        aoi_geojson=aoi,
        out_dir=tmp_path / "output",
        n_classes=4,
        classifier="kmeans",
        mmu_ha=0.0,
        smooth_radius_m=0.0,
        simplify_tolerance_m=0.0,
        export_format="gpkg",
        seed=42,
    )

    assert result["ok"] is True
    assert Path(result["vector"]).exists()
    assert result["metadata"]["classifier"] == "kmeans"
    assert result["metadata"]["n_classes"] == 4
    assert result["metadata"]["feature_count"] > 0


@pytest.mark.skip(reason="PyQGIS tests require QGIS system package installation")
def test_pyqgis_zones_quantiles_classification(tmp_path):
    """Test PyQGIS zones with quantiles classification (requires QGIS)."""
    from app.services.zones_pyqgis import build_zones_with_pyqgis

    ndvi_path = tmp_path / "ndvi.tif"
    _write_test_ndvi_raster(ndvi_path)

    aoi = {
        "type": "Polygon",
        "coordinates": [[[0, 0], [20, 0], [20, 20], [0, 20], [0, 0]]],
    }

    result = build_zones_with_pyqgis(
        ndvi_tif_path=ndvi_path,
        aoi_geojson=aoi,
        out_dir=tmp_path / "output",
        n_classes=5,
        classifier="quantiles",
        mmu_ha=0.5,
        smooth_radius_m=0.0,
        simplify_tolerance_m=0.0,
        export_format="gpkg",
        seed=42,
    )

    assert result["ok"] is True
    assert Path(result["vector"]).exists()
    assert result["metadata"]["classifier"] == "quantiles"
    assert result["metadata"]["n_classes"] == 5


@pytest.mark.skip(reason="PyQGIS tests require QGIS system package installation")
def test_pyqgis_zones_mmu_filter(tmp_path):
    """Test PyQGIS zones with MMU filtering (requires QGIS)."""
    from app.services.zones_pyqgis import build_zones_with_pyqgis

    ndvi_path = tmp_path / "ndvi.tif"
    _write_test_ndvi_raster(ndvi_path)

    aoi = {
        "type": "Polygon",
        "coordinates": [[[0, 0], [20, 0], [20, 20], [0, 20], [0, 0]]],
    }

    result = build_zones_with_pyqgis(
        ndvi_tif_path=ndvi_path,
        aoi_geojson=aoi,
        out_dir=tmp_path / "output",
        n_classes=4,
        classifier="kmeans",
        mmu_ha=1.0,  # 1 hectare MMU
        smooth_radius_m=0.0,
        simplify_tolerance_m=0.0,
        export_format="gpkg",
        seed=42,
    )

    assert result["ok"] is True
    assert result["metadata"]["mmu_ha"] == 1.0
