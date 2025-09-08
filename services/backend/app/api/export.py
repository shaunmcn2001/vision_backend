# Placeholder export endpoint code
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List
import zipfile, tempfile, os

router = APIRouter()

@router.post("/export")
async def export_geotiffs(years: List[int] = Form(...), file: UploadFile = File(...)):
    # Save uploaded file temporarily
    tmp_dir = tempfile.mkdtemp()
    input_path = os.path.join(tmp_dir, file.filename)
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Placeholder logic: just return an empty ZIP
    zip_path = os.path.join(tmp_dir, "output.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        zipf.writestr("README.txt", "GeoTIFF export coming soon.")

    return {"ok": True, "zip_file": zip_path}