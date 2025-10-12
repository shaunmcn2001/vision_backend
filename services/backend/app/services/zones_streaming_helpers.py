"""
Streaming helpers to keep 10 m resolution without out-of-memory.
Place this file at: services/backend/app/services/zones_streaming_helpers.py
Then make the two tiny edits in zones.py as shown in the instructions.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Tuple, List

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling

# Trigger streaming when rasters exceed this many pixels.
ZONES_STREAM_TRIGGER_PX = int(os.getenv("ZONES_STREAM_TRIGGER_PX", "60000000"))  # 60M px
# Keep default 10 m unless overridden by env. Use in zones.py: DEFAULT_SCALE = int(os.getenv("ZONES_SCALE_M", "10"))
# os.environ.setdefault("ZONES_SCALE_M", "10")  # optional set

def iter_windows(src: rasterio.DatasetReader, tile: int = 1024) -> Iterable[Window]:
    for row_off in range(0, src.height, tile):
        h = min(tile, src.height - row_off)
        for col_off in range(0, src.width, tile):
            w = min(tile, src.width - col_off)
            yield Window(col_off=col_off, row_off=row_off, width=w, height=h)

def compute_percentile_thresholds_stream(ndvi_path: Path, n_classes: int) -> np.ndarray:
    """Approximate percentiles using a streaming histogram over [-1, 1]."""
    if n_classes <= 1:
        return np.array([], dtype=np.float32)

    bins = 4096
    hist = np.zeros(bins, dtype=np.int64)
    lo, hi = -1.0, 1.0
    rng = (lo, hi)

    # Keep GDAL memory use conservative
    with rasterio.Env(GDAL_CACHEMAX="128"):
        with rasterio.open(ndvi_path) as src:
            nodata = src.nodata
            for win in iter_windows(src):
                block = src.read(1, window=win, out_dtype="float32")
                mask = np.isfinite(block)
                if nodata is not None:
                    mask &= block != nodata
                mask &= (block >= lo) & (block <= hi)
                if not mask.any():
                    continue
                vals = block[mask]
                h, _ = np.histogram(vals, bins=bins, range=rng)
                hist += h

    total = hist.sum()
    if total == 0:
        return np.array([], dtype=np.float32)

    cdf = np.cumsum(hist)
    percentiles = np.linspace(0, 100, n_classes + 1, dtype=np.float32)[1:-1]
    thresholds: list[float] = []
    for p in percentiles:
        target = (p / 100.0) * total
        idx = int(np.searchsorted(cdf, target, side="left"))
        idx = min(max(idx, 0), bins - 1)
        t = lo + (hi - lo) * (idx + 0.5) / bins
        thresholds.append(float(t))
    return np.array(thresholds, dtype=np.float32)

def classify_stream_to_file(
    ndvi_path: Path,
    thresholds: np.ndarray,
    out_path: Path,
) -> list[int]:
    """Classify NDVI into zones and write GeoTIFF window-by-window (uint8)."""
    with rasterio.open(ndvi_path) as src:
        profile = src.profile.copy()
        profile.update(dtype=rasterio.uint8, count=1, compress="lzw", nodata=0)
        unique_ids: set[int] = set()

        with rasterio.open(out_path, "w", **profile) as dst:
            for win in iter_windows(src):
                block = src.read(1, window=win, out_dtype="float32")
                mask = np.isfinite(block)
                if src.nodata is not None:
                    mask &= block != src.nodata
                mask &= (block >= -1.0) & (block <= 1.0)

                cls = np.zeros(block.shape, dtype=np.uint8)
                if thresholds.size:
                    idx = np.digitize(block[mask], thresholds, right=False) + 1  # 1..k
                    cls_masked = np.zeros(block[mask].shape, dtype=np.uint8)
                    cls_masked[:] = idx
                    cls[mask] = cls_masked
                    unique_ids.update(np.unique(idx).tolist())

                dst.write(cls, 1, window=win)

        return sorted(unique_ids)
