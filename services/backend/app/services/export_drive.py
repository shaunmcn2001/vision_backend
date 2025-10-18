
from __future__ import annotations
from typing import Optional, Dict, Any
import ee
from .earth_engine import ensure_ee

# Simple wrappers around EE Drive exports that return the task id.
# We create a tidy folder structure: <root>/<paddock>/<product>/

def _folder_path(root_folder: str, paddock: str, product: str) -> str:
    paddock = (paddock or "AOI").replace("/", "_")
    product = product.replace("/", "_")
    return f"{root_folder}/{paddock}/{product}"

def export_image_to_drive(
    image: ee.Image,
    region: ee.Geometry,
    filename_prefix: str,
    folder_root: str,
    paddock_name: str,
    product_name: str,
    scale: int = 10,
    crs: Optional[str] = None,
    file_format: str = "GeoTIFF"
) -> Dict[str, Any]:
    ensure_ee()
    folder = _folder_path(folder_root, paddock_name, product_name)
    desc = filename_prefix
    kwargs = dict(
        image=image,
        description=desc,
        folder=folder,
        fileNamePrefix=filename_prefix,
        region=region,
        scale=scale,
        maxPixels=1e13
    )
    if crs:
        kwargs["crs"] = crs
    if file_format:
        kwargs["fileFormat"] = file_format
    task = ee.batch.Export.image.toDrive(**kwargs)
    task.start()
    return {"task_id": task.id, "folder": folder, "name": filename_prefix, "type": "image"}

def export_table_to_drive(
    collection: ee.FeatureCollection,
    filename: str,
    folder_root: str,
    paddock_name: str,
    product_name: str,
    file_format: str = "SHP"  # SHP or CSV
) -> Dict[str, Any]:
    ensure_ee()
    folder = _folder_path(folder_root, paddock_name, product_name)
    desc = filename
    task = ee.batch.Export.table.toDrive(
        collection=collection,
        description=desc,
        folder=folder,
        fileFormat=file_format,
        fileNamePrefix=filename
    )
    task.start()
    return {"task_id": task.id, "folder": folder, "name": filename, "type": "table", "format": file_format}

def task_status(task_id: str) -> Dict[str, Any]:
    ensure_ee()
    task = ee.batch.Task.list()[0]  # dummy to init
    # EE Python API doesn't expose a direct get-by-id; scan list
    tasks = ee.batch.Task.list()
    for t in tasks:
        if t.id == task_id:
            return {"state": t.status().get("state"), "description": t.config.get("description")}
    return {"state": "UNKNOWN", "description": None}
