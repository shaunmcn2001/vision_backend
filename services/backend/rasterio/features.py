from typing import Iterable, Iterator, Tuple

import numpy as np

from .transform import Affine


def shapes(array: np.ndarray, mask: np.ndarray | None = None, transform: Affine | None = None) -> Iterator[Tuple[dict, int]]:
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
