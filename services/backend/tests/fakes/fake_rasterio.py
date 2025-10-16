"""Lightweight rasterio stand-in for tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterator, Tuple

import numpy as np

uint8 = np.uint8
int16 = np.int16
float32 = np.float32


@dataclass
class _Profile:
    width: int
    height: int
    count: int
    dtype: Any
    transform: Any
    crs: Any
    nodata: Any


class Dataset:
    def __init__(self, path: str | Path, mode: str, **kwargs: Any) -> None:
        self.path = Path(path)
        self.mode = mode
        self._closed = False
        if mode == "w":
            self._profile = _Profile(
                width=kwargs.get("width"),
                height=kwargs.get("height"),
                count=kwargs.get("count", 1),
                dtype=kwargs.get("dtype", np.float32),
                transform=kwargs.get("transform"),
                crs=kwargs.get("crs"),
                nodata=kwargs.get("nodata"),
            )
            shape = (self._profile.count, self._profile.height, self._profile.width)
            self._data = np.zeros(shape, dtype=self._profile.dtype)
            self._mask = np.full(shape, 255, dtype=np.uint8)
        elif mode == "r":
            payload = np.load(self.path, allow_pickle=True)
            data = payload["data"]
            mask = payload["mask"] if "mask" in payload.files else np.full_like(data, 255, dtype=np.uint8)
            profile_dict = json.loads(payload["profile"].item())
            self._data = data
            self._mask = mask
            transform_vals = profile_dict["transform"]
            transform = Affine(*transform_vals) if transform_vals else None
            self._profile = _Profile(
                width=profile_dict["width"],
                height=profile_dict["height"],
                count=profile_dict["count"],
                dtype=np.dtype(profile_dict["dtype"]),
                transform=transform,
                crs=profile_dict["crs"],
                nodata=profile_dict["nodata"],
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def __enter__(self) -> "Dataset":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        if self.mode == "w":
            profile_dict = {
                "width": self._profile.width,
                "height": self._profile.height,
                "count": self._profile.count,
                "dtype": str(np.dtype(self._profile.dtype)),
                "transform": list(self._profile.transform) if self._profile.transform else None,
                "crs": self._profile.crs,
                "nodata": self._profile.nodata,
            }
            with self.path.open("wb") as fh:
                np.savez(fh, data=self._data, mask=self._mask, profile=json.dumps(profile_dict))
        self._closed = True

    @property
    def transform(self) -> Any:
        return self._profile.transform

    @property
    def crs(self) -> Any:
        return self._profile.crs

    @property
    def nodata(self) -> Any:
        return self._profile.nodata

    @property
    def width(self) -> int:
        return self._profile.width

    @property
    def height(self) -> int:
        return self._profile.height

    @property
    def count(self) -> int:
        return self._profile.count

    @property
    def profile_dict(self) -> Dict[str, Any]:
        return {
            "width": self._profile.width,
            "height": self._profile.height,
            "count": self._profile.count,
            "dtype": np.dtype(self._profile.dtype).name,
            "transform": self._profile.transform,
            "crs": self._profile.crs,
            "nodata": self._profile.nodata,
        }

    @property
    def profile(self) -> Dict[str, Any]:
        return dict(self.profile_dict)

    @property
    def nodatavals(self) -> tuple[Any, ...]:
        return tuple(self._profile.nodata for _ in range(self._profile.count))

    def _slice_for_window(self, window: Any | None) -> tuple[slice, slice]:
        if window is None:
            return slice(None), slice(None)
        row_off = int(getattr(window, "row_off", 0))
        col_off = int(getattr(window, "col_off", 0))
        height = int(getattr(window, "height", self._profile.height))
        width = int(getattr(window, "width", self._profile.width))
        return slice(row_off, row_off + height), slice(col_off, col_off + width)

    def block_windows(self, index: int):  # type: ignore[override]
        yield (0, 0), Window(0, 0, self._profile.width, self._profile.height)

    def read(self, index: int, window: Any | None = None, masked: bool = False) -> np.ndarray:
        row_slice, col_slice = self._slice_for_window(window)
        data = np.array(self._data[index - 1, row_slice, col_slice], copy=True)
        mask = np.array(self._mask[index - 1, row_slice, col_slice], copy=True)
        if masked:
            mask_bool = mask == 0
            if self._profile.nodata is not None:
                mask_bool |= data == self._profile.nodata
            return np.ma.array(data, mask=mask_bool)
        return data

    def write(self, array: np.ndarray, index: int) -> None:
        self._data[index - 1] = array

    def write_mask(self, mask: np.ndarray) -> None:
        if mask.ndim == 2:
            expanded = np.broadcast_to(mask, (self._profile.count, *mask.shape))
        else:
            expanded = mask
        self._mask[:] = expanded

    def read_masks(self, index: int, window: Any | None = None) -> np.ndarray:
        row_slice, col_slice = self._slice_for_window(window)
        return np.array(self._mask[index - 1, row_slice, col_slice], copy=True)


def open(path: str | Path, mode: str = "r", **kwargs: Any) -> Dataset:
    return Dataset(path, mode, **kwargs)


class Affine:
    def __init__(self, a: float, b: float, c: float, d: float, e: float, f: float) -> None:
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f

    def __call__(self, col: float, row: float) -> tuple[float, float]:
        x = self.a * col + self.b * row + self.c
        y = self.d * col + self.e * row + self.f
        return x, y

    def __iter__(self):  # type: ignore[override]
        yield from (self.a, self.b, self.c, self.d, self.e, self.f)

    @classmethod
    def identity(cls) -> "Affine":
        return cls(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)


def from_origin(west: float, north: float, xsize: float, ysize: float) -> Affine:
    return Affine(xsize, 0.0, west, 0.0, -ysize, north)


class Window:
    def __init__(self, col_off: int, row_off: int, width: int, height: int) -> None:
        self.col_off = col_off
        self.row_off = row_off
        self.width = width
        self.height = height

    def flatten(self) -> tuple[int, int, int, int]:
        return (
            self.row_off,
            self.row_off + self.height,
            self.col_off,
            self.col_off + self.width,
        )


windows = ModuleType("rasterio.windows")
windows.Window = Window


def shapes(
    array: np.ndarray,
    mask: np.ndarray | None = None,
    transform: Affine | None = None,
) -> Iterator[Tuple[dict, int]]:
    if mask is None:
        mask = np.ones_like(array, dtype=bool)
    rows, cols = array.shape
    transform = transform or Affine.identity()

    for row in range(rows):
        for col in range(cols):
            if not mask[row, col]:
                continue
            value = array[row, col]
            if np.isnan(value):
                continue
            x0, y0 = transform(col, row)
            x1, y1 = transform(col + 1, row + 1)
            polygon = {
                "type": "Polygon",
                "coordinates": [
                    [
                        [x0, y0],
                        [x0, y1],
                        [x1, y1],
                        [x1, y0],
                        [x0, y0],
                    ]
                ],
            }
            yield polygon, value


transform = ModuleType("rasterio.transform")
transform.Affine = Affine
transform.from_origin = from_origin

features = ModuleType("rasterio.features")
features.shapes = shapes

__all__ = [
    "uint8",
    "int16",
    "float32",
    "Dataset",
    "open",
    "Affine",
    "from_origin",
    "shapes",
    "transform",
    "features",
]
