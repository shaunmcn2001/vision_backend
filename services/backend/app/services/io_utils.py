from __future__ import annotations

import os
import pathlib
import shutil
import zipfile
from typing import Iterable

import requests

CHUNK = int(os.getenv("EXPORT_CHUNK_MB", "1")) * 1024 * 1024


def stream_to_file(url: str, out_path: str, timeout: int = 300) -> None:
    """Stream a remote resource to disk without buffering the entire payload."""
    target = pathlib.Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with target.open("wb") as handle:
            shutil.copyfileobj(response.raw, handle, length=CHUNK)


def zip_paths(zip_path: str, paths: Iterable[str]) -> None:
    """Zip a collection of paths from disk using streaming writes."""
    target = pathlib.Path(zip_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in paths:
            archive.write(path, arcname=pathlib.Path(path).name)


__all__ = ["stream_to_file", "zip_paths"]
