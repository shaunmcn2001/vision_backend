import io
import sys
import tempfile
import zipfile
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi import HTTPException

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

from app.utils.shapefile import shapefile_zip_to_geojson


def test_malicious_zip_rejected():
    unique_name = f"vision_backend_{uuid4().hex}.shp"
    target_file = Path(tempfile.gettempdir()) / unique_name
    if target_file.exists():
        target_file.unlink()

    malicious_member = target_file.as_posix()

    with io.BytesIO() as buffer:
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr(malicious_member, b"dummy")
        payload = buffer.getvalue()

    with pytest.raises(HTTPException) as excinfo:
        shapefile_zip_to_geojson(payload)

    assert excinfo.value.status_code == 400
    assert "unsafe" in excinfo.value.detail.lower()
    assert not target_file.exists()
