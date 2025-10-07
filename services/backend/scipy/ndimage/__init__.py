import numpy as np


def generate_binary_structure(rank: int, connectivity: int) -> np.ndarray:
    if rank != 2:
        raise ValueError("Only 2D structures are supported in this stub")
    return np.ones((3, 3), dtype=int)


def label(input_array: np.ndarray, structure: np.ndarray | None = None):
    data = np.asarray(input_array).astype(bool)
    labeled = np.zeros_like(data, dtype=int)
    current = 0
    rows, cols = data.shape

    for row in range(rows):
        for col in range(cols):
            if not data[row, col] or labeled[row, col] != 0:
                continue
            current += 1
            stack = [(row, col)]
            while stack:
                r, c = stack.pop()
                if (
                    r < 0
                    or r >= rows
                    or c < 0
                    or c >= cols
                    or not data[r, c]
                    or labeled[r, c] != 0
                ):
                    continue
                labeled[r, c] = current
                stack.extend(
                    [
                        (r - 1, c),
                        (r + 1, c),
                        (r, c - 1),
                        (r, c + 1),
                    ]
                )
    return labeled, current


def sum(input_array: np.ndarray, labels: np.ndarray, index) -> np.ndarray:
    data = np.asarray(input_array)
    label_data = np.asarray(labels)
    if np.isscalar(index):
        indices = [index]
    else:
        indices = list(index)
    results = []
    for idx in indices:
        results.append(data[label_data == idx].sum())
    return np.array(results)


def generic_filter(
    array: np.ndarray, function, size: int, mode: str = "nearest"
) -> np.ndarray:
    data = np.asarray(array)
    pad = size // 2
    if mode != "nearest":
        raise ValueError("Only nearest mode is supported in this stub")
    padded = np.pad(data, pad, mode="edge")
    result = np.zeros_like(data, dtype=float)
    rows, cols = data.shape
    for r in range(rows):
        for c in range(cols):
            window = padded[r : r + size, c : c + size]
            result[r, c] = function(window)
    return result
