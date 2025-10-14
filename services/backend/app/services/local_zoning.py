from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional
import json
import numpy as np
import rasterio
from rasterio.features import shapes
import geopandas as gpd

def _majority_filter(arr: np.ndarray, iterations: int = 1) -> np.ndarray:
    h, w = arr.shape
    out = arr.copy()
    for _ in range(iterations):
        new = out.copy()
        # 3x3 majority per pixel (naive but dependency-free)
        for y in range(1, h-1):
            for x in range(1, w-1):
                block = out[y-1:y+2, x-1:x+2].ravel()
                vals, counts = np.unique(block[block>0], return_counts=True)
                if counts.size:
                    new[y,x] = vals[counts.argmax()]
        out = new
    return out

def classify_ndvi_tif(ndvi_tif: Path, out_raster: Path, n_zones: int = 5, smooth_iters: int = 1, nodata: Optional[float] = None) -> Tuple[Path, List[float]]:
    out_raster = Path(out_raster).with_suffix(".tif")
    with rasterio.open(ndvi_tif) as src:
        ndvi = src.read(1, masked=False)
        profile = src.profile.copy()
        if nodata is None:
            nodata = profile.get("nodata", None)
        mask = np.isfinite(ndvi)
        if nodata is not None:
            mask &= ~np.isclose(ndvi, nodata)
        valid = ndvi[mask]
        if valid.size == 0:
            cls = np.zeros_like(ndvi, dtype=np.int16)
            thresholds: List[float] = []
        else:
            qs = np.linspace(0, 100, n_zones + 1)[1:-1]
            thresholds = np.percentile(valid, qs).tolist()
            cls = np.zeros_like(ndvi, dtype=np.int16)
            prev = -np.inf
            for i, thr in enumerate(thresholds, start=1):
                cls[(ndvi > prev) & (ndvi <= thr)] = i
                prev = thr
            cls[ndvi > prev] = n_zones
            if smooth_iters > 0:
                cls = _majority_filter(cls, iterations=smooth_iters)
        profile.update(dtype=rasterio.int16, count=1, nodata=0, compress="DEFLATE")
        with rasterio.open(out_raster, "w", **profile) as dst:
            dst.write(cls, 1)
    return out_raster, thresholds

def polygonize_zones(zones_tif: Path, out_geojson: Path, simplify_tolerance: float = 0.0) -> Path:
    out_geojson = Path(out_geojson).with_suffix(".geojson")
    with rasterio.open(zones_tif) as src:
        arr = src.read(1)
        transform = src.transform
        crs = src.crs
        feats = [{"properties": {"zone": int(v)}, "geometry": g}
                 for g, v in shapes(arr, mask=None, transform=transform) if v and int(v) > 0]
        if not feats:
            out_geojson.write_text(json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8")
            return out_geojson
        gdf = gpd.GeoDataFrame.from_features(feats, crs=crs)
        if simplify_tolerance > 0:
            gdf["geometry"] = gdf.geometry.simplify(simplify_tolerance, preserve_topology=True)
        gdf.to_file(out_geojson, driver="GeoJSON")
    return out_geojson

def write_kml(geojson_path: Path, out_kml: Path) -> Path:
    out_kml = Path(out_kml).with_suffix(".kml")
    gdf = gpd.read_file(geojson_path)
    gdf.to_file(out_kml, driver="KML")
    return out_kml
