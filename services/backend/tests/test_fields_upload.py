import io
import sys
import textwrap
import zipfile
from pathlib import Path

import pytest
from fastapi import UploadFile
from shapely.geometry import shape

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

from app.api import fields_upload  # noqa: E402


@pytest.fixture
def anyio_backend():
    return "asyncio"


KML_POLYGON = (
    textwrap.dedent(
        """\
    <?xml version="1.0" encoding="UTF-8"?>
    <kml xmlns="http://www.opengis.net/kml/2.2">
      <Document>
        <Placemark>
          <name>Test polygon</name>
          <Polygon>
            <outerBoundaryIs>
              <LinearRing>
                <coordinates>
                  130.0,-25.0,0 130.0,-25.01,0 130.01,-25.01,0 130.01,-25.0,0 130.0,-25.0,0
                </coordinates>
              </LinearRing>
            </outerBoundaryIs>
          </Polygon>
        </Placemark>
      </Document>
    </kml>
    """
    )
    .strip()
    .encode("utf-8")
)


def _payload(is_kmz: bool) -> bytes:
    if not is_kmz:
        return KML_POLYGON
    with io.BytesIO() as buffer:
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr("doc.kml", KML_POLYGON)
        return buffer.getvalue()


@pytest.mark.parametrize("is_kmz", [False, True])
def test_kml_kmz_to_geojson_accepts_polygon(is_kmz):
    payload = _payload(is_kmz)
    geom_mapping = fields_upload._kml_or_kmz_to_geojson(payload, is_kmz=is_kmz)
    geom = shape(geom_mapping)

    assert geom.geom_type == "MultiPolygon"
    assert geom.is_valid
    assert not geom.is_empty
    assert geom.area > 0


@pytest.mark.anyio("asyncio")
@pytest.mark.parametrize("suffix,is_kmz", [(".kml", False), (".kmz", True)])
async def test_upload_field_accepts_polygon(monkeypatch, suffix, is_kmz):
    stored = {}

    def fake_upload_json(data, path, content_type="application/json"):
        stored[path] = {"data": data, "content_type": content_type}
        return path

    def fake_download_json(path):
        return stored[path]["data"]

    def fake_exists(path):
        return path in stored

    monkeypatch.setattr(fields_upload, "upload_json", fake_upload_json)
    monkeypatch.setattr(fields_upload, "download_json", fake_download_json)
    monkeypatch.setattr(fields_upload, "exists", fake_exists)
    monkeypatch.setattr(fields_upload, "MIN_FIELD_HA", 0.0)

    dummy_hex = "1234567890abcdef"
    monkeypatch.setattr(
        fields_upload,
        "uuid4",
        lambda: type("DummyUUID", (), {"hex": dummy_hex})(),
    )

    upload = UploadFile(filename=f"test{suffix}", file=io.BytesIO(_payload(is_kmz)))
    response = await fields_upload.upload_field(upload, name="Example field")

    assert response["ok"] is True

    field_id = dummy_hex[:12]
    saved_geom = stored[f"fields/{field_id}/field.geojson"]["data"]
    saved_shape = shape(saved_geom)
    assert saved_shape.is_valid
    assert not saved_shape.is_empty
    assert saved_shape.area > 0
