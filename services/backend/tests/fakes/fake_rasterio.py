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
        elif mode == "r":
            payload = np.load(self.path, allow_pickle=True)
            data = payload["data"]
            profile_dict = json.loads(payload["profile"].item())
            self._data = data
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
                np.savez(fh, data=self._data, profile=json.dumps(profile_dict))
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

    def read(self, index: int, masked: bool = False) -> np.ndarray:
        data = np.array(self._data[index - 1], copy=True)
        if masked:
            mask = np.zeros_like(data, dtype=bool)
            if self._profile.nodata is not None:
                mask |= data == self._profile.nodata
            return np.ma.array(data, mask=mask)
        return data

    def write(self, array: np.ndarray, index: int) -> None:
        self._data[index - 1] = array


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
