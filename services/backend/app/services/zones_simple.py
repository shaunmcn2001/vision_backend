# services/backend/app/services/zones_simple.py
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid
import rasterio
from rasterio.features import shapes
from rasterio.transform import Affine
from rasterio.enums import Resampling
from fiona import open as fiona_open
from fiona.crs import CRS

logger = logging.getLogger(__name__)

# ----------------------------- datatypes ------------------------------------


@dataclass
class SimpleZonesParams:
    n_classes: int = 5  # 3..7 typical
    classifier: str = "quantiles"  # "quantiles" | "kmeans"
    mmu_ha: float = 2.0  # minimum mapping unit (hectares)
    simplify_tol_m: float = 5.0  # Douglas-Peucker simplify tolerance, meters
    output_format: str = "gpkg"  # "gpkg" | "geojson" | "shp"
    seed: int = 42


# ---------------------------- helpers ---------------------------------------


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _pixel_area_m2(transform: Affine) -> float:
    # approximate square pixel area (handles north-up georeferencing)
    sx = abs(transform.a)
    sy = abs(transform.e)
    return sx * sy


def _raster_read_single_band(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    with rasterio.open(path) as ds:
        arr = ds.read(1, out_dtype="float32", resampling=Resampling.nearest)
        nodata = ds.nodata
        transform = ds.transform
        crs = ds.crs
        return arr, {
            "transform": transform,
            "crs": crs,
            "nodata": nodata,
            "width": ds.width,
            "height": ds.height,
        }


def _classify_quantiles(
    ndvi: np.ndarray, nodata: Optional[float], n: int
) -> Tuple[np.ndarray, List[float]]:
    mask = np.isfinite(ndvi)
    if nodata is not None:
        mask &= ~np.isclose(ndvi, nodata)
    vals = ndvi[mask]
    if vals.size == 0:
        raise RuntimeError("E_RANGE_EMPTY: no valid NDVI values.")
    qs = np.linspace(0.0, 1.0, n + 1)
    brks = np.quantile(vals, qs)
    brks = np.unique(brks)
    if brks.size <= 2:
        raise RuntimeError("E_BREAKS_COLLAPSED: NDVI distribution collapsed.")
    bins = np.digitize(vals, brks[1:-1], right=False)
    labels = np.zeros_like(ndvi, dtype=np.int32)
    labels[mask] = bins + 1  # classes 1..n
    return labels, brks.tolist()


def _classify_kmeans(
    ndvi: np.ndarray, nodata: Optional[float], n: int, seed: int
) -> Tuple[np.ndarray, List[float]]:
    from sklearn.cluster import KMeans

    mask = np.isfinite(ndvi)
    if nodata is not None:
        mask &= ~np.isclose(ndvi, nodata)
    vals = ndvi[mask].reshape(-1, 1)
    if vals.size == 0:
        raise RuntimeError("E_RANGE_EMPTY: no valid NDVI values.")
    # tiny jitter to avoid ties
    rng = np.random.RandomState(seed)
    vals = vals + rng.randn(*vals.shape) * 1e-6
    km = KMeans(n_clusters=n, n_init="auto", random_state=seed)
    km.fit(vals)
    labels = np.zeros_like(ndvi, dtype=np.int32)
    labels[mask] = km.labels_ + 1
    # compute class-wise NDVI medians as pseudo-breaks for reporting
    medians = []
    for k in range(n):
        med = (
            np.median(vals[km.labels_ == k])
            if np.any(km.labels_ == k)
            else float("nan")
        )
        medians.append(float(med))
    return labels, medians


def _polygonize_classes(
    class_grid: np.ndarray, transform: Affine, crs
) -> List[Dict[str, Any]]:
    # rasterio.features.shapes to emit polygons by integer class
    mask = class_grid > 0
    results = []
    for geom, val in shapes(
        class_grid.astype(np.int32), mask=mask, transform=transform, connectivity=4
    ):
        cls = int(val)
        if cls <= 0:
            continue
        geom_s = shape(geom)
        geom_s = make_valid(geom_s)
        if geom_s.is_empty:
            continue
        # normalize to MultiPolygon
        if isinstance(geom_s, Polygon):
            geom_s = MultiPolygon([geom_s])
        results.append({"geometry": geom_s, "class": cls})
    return results


def _apply_mmu_and_simplify(
    features: List[Dict[str, Any]],
    px_area_m2: float,
    mmu_ha: float,
    simplify_tol_m: float,
) -> List[Dict[str, Any]]:
    mmu_m2 = mmu_ha * 10000.0
    out = []
    for f in features:
        geom: MultiPolygon = f["geometry"]
        # drop tiny rings by area
        filtered_polys = [p for p in geom.geoms if p.area >= mmu_m2]
        if not filtered_polys:
            continue
        merged = unary_union(filtered_polys)
        if simplify_tol_m and simplify_tol_m > 0:
            merged = merged.simplify(simplify_tol_m, preserve_topology=True)
        if merged.is_empty:
            continue
        if isinstance(merged, Polygon):
            merged = MultiPolygon([merged])
        out.append({"geometry": merged, "class": f["class"]})
    return out


def _write_vector(features: List[Dict[str, Any]], crs, out_path: str) -> str:
    _ensure_dir(os.path.dirname(out_path))
    driver = (
        "GPKG"
        if out_path.lower().endswith(".gpkg")
        else ("GeoJSON" if out_path.lower().endswith(".geojson") else "ESRI Shapefile")
    )
    schema = {"geometry": "MultiPolygon", "properties": {"class": "int"}}
    crs_fiona = CRS.from_user_input(
        crs.to_string() if hasattr(crs, "to_string") else str(crs)
    )
    with fiona_open(out_path, "w", driver=driver, schema=schema, crs=crs_fiona) as dst:
        for f in features:
            dst.write(
                {
                    "geometry": mapping(f["geometry"]),
                    "properties": {"class": int(f["class"])},
                }
            )
    return out_path


# --------------------------- main entrypoints -------------------------------


def build_zones_simple_from_ndvi_tif(
    ndvi_tif: str,
    *,
    params: SimpleZonesParams,
    out_dir: str,
) -> Dict[str, Any]:
    ndvi, meta = _raster_read_single_band(ndvi_tif)
    nodata = meta["nodata"]
    transform: Affine = meta["transform"]
    crs = meta["crs"]
    px_area = _pixel_area_m2(transform)

    vmin, vmax = float(np.nanmin(ndvi)), float(np.nanmax(ndvi))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        raise RuntimeError("E_RANGE_EMPTY: NDVI has no dynamic range.")

    if params.classifier == "kmeans":
        classes, breaks = _classify_kmeans(ndvi, nodata, params.n_classes, params.seed)
    else:
        classes, breaks = _classify_quantiles(ndvi, nodata, params.n_classes)

    feats = _polygonize_classes(classes, transform, crs)
    feats = _apply_mmu_and_simplify(
        feats, px_area, params.mmu_ha, params.simplify_tol_m
    )

    ext = {"gpkg": ".gpkg", "geojson": ".geojson", "shp": ".shp"}[params.output_format]
    out_vec = os.path.join(out_dir, f"zones_simple{ext}")
    path = _write_vector(feats, crs, out_vec)

    return {
        "ok": True,
        "vector": path,
        "metadata": {
            "classifier": params.classifier,
            "n_classes": params.n_classes,
            "breaks": breaks,
            "mmu_ha": params.mmu_ha,
            "simplify_tol_m": params.simplify_tol_m,
        },
    }


# Optional: Earth Engine → single mean NDVI GeoTIFF then local cartography
def build_zones_simple_from_ee(
    aoi_geojson: Dict[str, Any],
    start_date: str,
    end_date: str,
    *,
    params: SimpleZonesParams,
    out_dir: str,
    bucket: Optional[str] = None,
    gcs_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Minimal EE usage: compute 1-b and mean NDVI, export once to local GeoTIFF or GCS, then do local cartography.
    NOTE: Implement your existing EE export→download method here; this stub expects a local ndvi_mean.tif to exist.
    """
    # Keeping this minimal: use your existing NDVI export utility to produce a local tif:
    # ndvi_tif = ensure_mean_ndvi_local_tif(aoi_geojson, start_date, end_date, bucket, gcs_prefix)
    raise NotImplementedError(
        "Plug your existing EE NDVI export here, then call build_zones_simple_from_ndvi_tif"
    )
