
import ee
from pathlib import Path
from urllib.request import urlopen

# --- Vector download helper ---
def _download_vector_to_path(vectors: ee.FeatureCollection, target: Path, *, file_format: str) -> None:
    fmt = (file_format or "geojson").lower()
    ee_fmt_map = {
        "geojson": "GEO_JSON",
        "json": "GEO_JSON",
        "kml": "KML",
        "kmz": "KMZ",
        "shp": "SHP",
        "shapefile": "SHP",
        "csv": "CSV",
    }
    ee_fmt = ee_fmt_map.get(fmt, "GEO_JSON")

    params = {
        "filetype": ee_fmt,
        "selectors": ["zone"],
        "filename": target.stem,
    }

    if ee_fmt == "GEO_JSON":
        target = target.with_suffix(".geojson")
    elif ee_fmt in ("KML", "KMZ"):
        target = target.with_suffix("." + fmt)
    elif ee_fmt == "CSV":
        target = target.with_suffix(".csv")
    elif ee_fmt == "SHP":
        target = target.with_suffix(".zip")

    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        url = vectors.getDownloadURL(**params)
    except Exception as e:
        raise ValueError(f"EE getDownloadURL failed: {e}")

    with urlopen(url) as response:
        data = response.read()
        target.write_bytes(data)


# --- NDVI export block ---
def export_ndvi_layers(ndvi_mean_native, classified_image, geometry, workdir, _download_image_to_path, _DownloadParams, DEFAULT_EXPORT_CRS, DEFAULT_SCALE):
    # Mean NDVI (export-friendly)
    ndvi_path = workdir / "NDVI_mean.tif"
    ndvi_mean_export_img = (
        ee.Image(ndvi_mean_native)
          .rename("NDVI_mean")
          .toFloat()
          .unmask(-9999)
          .clip(geometry)
    )
    mean_export = _download_image_to_path(
        ndvi_mean_export_img,
        geometry,
        ndvi_path,
        params=_DownloadParams(crs=DEFAULT_EXPORT_CRS, scale=DEFAULT_SCALE),
    )
    ndvi_path = mean_export.path

    # Classified zones (int8)
    classified_path = workdir / "zones.tif"
    classified_export_img = (
        ee.Image(classified_image)
          .rename("zone")
          .toInt8()
          .unmask(0)
          .clip(geometry)
    )
    classified_export = _download_image_to_path(
        classified_export_img,
        geometry,
        classified_path,
        params=_DownloadParams(crs=DEFAULT_EXPORT_CRS, scale=DEFAULT_SCALE),
    )
    classified_path = classified_export.path
    return ndvi_path, classified_path


# --- Polygonization reducer fix ---
def polygonize_zones(cls_mmu, geom):
    vectors = cls_mmu.reduceToVectors(
        geometry=geom,
        scale=10,
        geometryType="polygon",
        labelProperty="zone",
        reducer=ee.Reducer.mode(),
        bestEffort=True,
        maxPixels=1e9,
    )
    return vectors
