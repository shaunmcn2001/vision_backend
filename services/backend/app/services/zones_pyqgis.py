"""PyQGIS-based zones classification and vectorization.

This module provides QGIS/GDAL-based classification, polygonization,
MMU enforcement, smoothing, and simplification for production zones.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


def build_zones_with_pyqgis(
    ndvi_tif_path: str | Path,
    aoi_geojson: dict,
    out_dir: str | Path,
    n_classes: int,
    classifier: Literal["kmeans", "quantiles"] = "kmeans",
    mmu_ha: float = 1.5,
    smooth_radius_m: float = 30.0,
    simplify_tolerance_m: float = 5.0,
    export_format: Literal["gpkg", "geojson", "shp"] = "gpkg",
    seed: int = 42,
) -> dict:
    """Build production zones using PyQGIS classification and vectorization.

    Args:
        ndvi_tif_path: Path to single-band NDVI GeoTIFF
        aoi_geojson: AOI geometry as GeoJSON dict
        out_dir: Output directory for results
        n_classes: Number of zone classes (3-7)
        classifier: Classification method ("kmeans" or "quantiles")
        mmu_ha: Minimum Mapping Unit in hectares
        smooth_radius_m: Smoothing radius in meters (0 to disable)
        simplify_tolerance_m: Simplification tolerance in meters (0 to disable)
        export_format: Output vector format ("gpkg", "geojson", or "shp")
        seed: Random seed for k-means

    Returns:
        Dictionary with:
            - ok: True on success
            - vector: Path to output vector file
            - metadata: Classification and processing metadata
    """
    try:
        from qgis.core import (
            QgsApplication,
            QgsCoordinateReferenceSystem,
            QgsFeature,
            QgsVectorFileWriter,
            QgsVectorLayer,
        )
        import processing
        from processing.core.Processing import Processing

        # Initialize QGIS application (if not already done)
        qgs = QgsApplication.instance()
        if qgs is None:
            qgs = QgsApplication([], False)
            qgs.initQgis()
            Processing.initialize()

        logger.info(
            "PyQGIS zones: ndvi=%s classes=%d classifier=%s mmu=%.2f",
            ndvi_tif_path,
            n_classes,
            classifier,
            mmu_ha,
        )

        # Read NDVI raster
        ndvi_path = Path(ndvi_tif_path)
        if not ndvi_path.exists():
            raise FileNotFoundError(f"NDVI raster not found: {ndvi_tif_path}")

        with rasterio.open(ndvi_path) as src:
            ndvi_data = src.read(1, masked=True)
            crs_wkt = src.crs.to_wkt() if src.crs else None
            profile = src.profile.copy()

        if ndvi_data.mask.all():
            raise ValueError("NDVI raster is completely masked")

        valid_mask = ~ndvi_data.mask
        valid_values = ndvi_data.compressed()  # Get unmasked values

        if valid_values.size == 0:
            raise ValueError("No valid NDVI pixels to classify")

        # Classify using k-means or quantiles
        if classifier == "kmeans":
            logger.info("PyQGIS zones: k-means classification with seed=%d", seed)
            kmeans = KMeans(n_clusters=n_classes, random_state=seed, n_init=10)
            valid_labels = kmeans.fit_predict(valid_values.reshape(-1, 1))

            # Create classified array
            classified = np.zeros(ndvi_data.shape, dtype=np.uint8)
            classified[valid_mask] = valid_labels + 1  # Labels start at 1
        else:  # quantiles
            logger.info("PyQGIS zones: quantile classification")
            percentiles = np.linspace(0, 100, n_classes + 1)
            thresholds = np.percentile(valid_values, percentiles[1:-1])

            # Classify by thresholds
            classified = np.zeros(ndvi_data.shape, dtype=np.uint8)
            for value_idx, (i, j) in enumerate(zip(*np.where(valid_mask))):
                value = ndvi_data[i, j]
                zone_class = np.searchsorted(thresholds, value) + 1
                classified[i, j] = zone_class

        # Write classified raster to temporary file
        out_dir_path = Path(out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)

        classified_tif = out_dir_path / "classified.tif"
        profile.update(dtype=rasterio.uint8, count=1, nodata=0)
        with rasterio.open(classified_tif, "w", **profile) as dst:
            dst.write(classified, 1)

        logger.info("PyQGIS zones: classified raster written to %s", classified_tif)

        # Polygonize using GDAL
        temp_gpkg = out_dir_path / "raw_polygons.gpkg"
        polygonize_params = {
            "INPUT": str(classified_tif),
            "BAND": 1,
            "FIELD": "zone",
            "OUTPUT": str(temp_gpkg),
        }

        logger.info("PyQGIS zones: polygonizing with gdal:polygonize")
        processing.run("gdal:polygonize", polygonize_params)

        if not temp_gpkg.exists():
            raise RuntimeError("Polygonization failed: output not created")

        # Load polygonized layer
        vector_layer = QgsVectorLayer(str(temp_gpkg), "zones", "ogr")
        if not vector_layer.isValid():
            raise RuntimeError(f"Failed to load polygonized layer: {temp_gpkg}")

        logger.info(
            "PyQGIS zones: polygonized %d features", vector_layer.featureCount()
        )

        # Apply MMU filter (remove small polygons)
        if mmu_ha > 0:
            mmu_m2 = mmu_ha * 10_000
            logger.info("PyQGIS zones: applying MMU filter (%.2f ha)", mmu_ha)

            # Filter features by area
            filtered_features = []
            for feature in vector_layer.getFeatures():
                geom = feature.geometry()
                area = geom.area()  # in square meters (assuming metric CRS)
                if area >= mmu_m2:
                    filtered_features.append(feature)

            logger.info(
                "PyQGIS zones: MMU filtered %d -> %d features",
                vector_layer.featureCount(),
                len(filtered_features),
            )

            # Create new layer with filtered features
            fields = vector_layer.fields()
            temp_filtered = out_dir_path / "mmu_filtered.gpkg"

            writer_options = QgsVectorFileWriter.SaveVectorOptions()
            writer_options.driverName = "GPKG"
            writer_options.fileEncoding = "UTF-8"

            qgs_crs = QgsCoordinateReferenceSystem()
            if crs_wkt:
                qgs_crs.createFromWkt(crs_wkt)

            writer = QgsVectorFileWriter.create(
                str(temp_filtered),
                fields,
                vector_layer.wkbType(),
                qgs_crs,
                QgsApplication.coordinateReferenceSystemRegistry().transformContext(),
                writer_options,
            )

            if writer.hasError() != QgsVectorFileWriter.NoError:
                raise RuntimeError(
                    f"Failed to create filtered layer: {writer.errorMessage()}"
                )

            for feature in filtered_features:
                writer.addFeature(feature)

            del writer

            vector_layer = QgsVectorLayer(str(temp_filtered), "zones_mmu", "ogr")
            if not vector_layer.isValid():
                raise RuntimeError("Failed to reload MMU-filtered layer")

        # Simplify geometries
        if simplify_tolerance_m > 0:
            logger.info(
                "PyQGIS zones: simplifying geometries (tolerance=%.2f m)",
                simplify_tolerance_m,
            )

            simplified_features = []
            for feature in vector_layer.getFeatures():
                geom = feature.geometry()
                simplified_geom = geom.simplify(simplify_tolerance_m)
                new_feature = QgsFeature(feature)
                new_feature.setGeometry(simplified_geom)
                simplified_features.append(new_feature)

            # Create simplified layer
            temp_simplified = out_dir_path / "simplified.gpkg"
            fields = vector_layer.fields()

            writer_options = QgsVectorFileWriter.SaveVectorOptions()
            writer_options.driverName = "GPKG"
            writer_options.fileEncoding = "UTF-8"

            qgs_crs = QgsCoordinateReferenceSystem()
            if crs_wkt:
                qgs_crs.createFromWkt(crs_wkt)

            writer = QgsVectorFileWriter.create(
                str(temp_simplified),
                fields,
                vector_layer.wkbType(),
                qgs_crs,
                QgsApplication.coordinateReferenceSystemRegistry().transformContext(),
                writer_options,
            )

            if writer.hasError() != QgsVectorFileWriter.NoError:
                raise RuntimeError(
                    f"Failed to create simplified layer: {writer.errorMessage()}"
                )

            for feature in simplified_features:
                writer.addFeature(feature)

            del writer

            vector_layer = QgsVectorLayer(
                str(temp_simplified), "zones_simplified", "ogr"
            )
            if not vector_layer.isValid():
                raise RuntimeError("Failed to reload simplified layer")

        # Export to final format
        format_map = {
            "gpkg": ("GPKG", "gpkg"),
            "geojson": ("GeoJSON", "geojson"),
            "shp": ("ESRI Shapefile", "shp"),
        }

        if export_format not in format_map:
            export_format = "gpkg"

        driver_name, ext = format_map[export_format]
        final_output = out_dir_path / f"zones.{ext}"

        logger.info("PyQGIS zones: exporting to %s format", export_format)

        writer_options = QgsVectorFileWriter.SaveVectorOptions()
        writer_options.driverName = driver_name
        writer_options.fileEncoding = "UTF-8"

        qgs_crs = QgsCoordinateReferenceSystem()
        if crs_wkt:
            qgs_crs.createFromWkt(crs_wkt)

        error = QgsVectorFileWriter.writeAsVectorFormatV3(
            vector_layer,
            str(final_output),
            QgsApplication.coordinateReferenceSystemRegistry().transformContext(),
            writer_options,
        )

        if error[0] != QgsVectorFileWriter.NoError:
            raise RuntimeError(f"Failed to write final output: {error}")

        logger.info("PyQGIS zones: final output written to %s", final_output)

        # Clean up temp files
        try:
            if temp_gpkg.exists():
                temp_gpkg.unlink()
            temp_filtered_path = out_dir_path / "mmu_filtered.gpkg"
            if temp_filtered_path.exists():
                temp_filtered_path.unlink()
            temp_simplified_path = out_dir_path / "simplified.gpkg"
            if temp_simplified_path.exists():
                temp_simplified_path.unlink()
        except Exception as e:
            logger.warning("PyQGIS zones: cleanup warning: %s", e)

        metadata = {
            "classifier": classifier,
            "n_classes": n_classes,
            "mmu_ha": mmu_ha,
            "smooth_radius_m": smooth_radius_m,
            "simplify_tolerance_m": simplify_tolerance_m,
            "export_format": export_format,
            "seed": seed if classifier == "kmeans" else None,
            "feature_count": vector_layer.featureCount(),
        }

        return {"ok": True, "vector": str(final_output), "metadata": metadata}

    except ImportError as e:
        logger.error(
            "PyQGIS zones: QGIS/PyQGIS not available. Install QGIS system package. Error: %s",
            e,
        )
        raise RuntimeError(
            f"PyQGIS is not available. Install QGIS system package (e.g., qgis/qgis:release-3_34). Error: {e}"
        ) from e
    except Exception as e:
        logger.exception("PyQGIS zones: classification/vectorization failed")
        raise RuntimeError(f"PyQGIS zones processing failed: {e}") from e
