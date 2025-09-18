from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional, List
import os, io, zipfile, json
from fastkml import kml
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from app.services.gcs import upload_json, download_json, exists
from uuid import uuid4
from datetime import datetime, timezone
from app.utils.shapefile import as_multipolygon, shapefile_zip_to_geojson
from app.utils.geometry import area_ha

router = APIRouter()

MIN_FIELD_HA = float(os.getenv("MIN_FIELD_HA", "1.0"))  # default 1 ha
def _kml_or_kmz_to_geojson(file_bytes: bytes, is_kmz: bool) -> dict:
    """
    Extract polygons from KML/KMZ → MultiPolygon (EPSG:4326).
    """
    k = kml.KML()
    data = file_bytes
    if is_kmz:
        with zipfile.ZipFile(io.BytesIO(file_bytes), "r") as z:
            # find the first .kml inside KMZ
            kml_name = next((n for n in z.namelist() if n.lower().endswith(".kml")), None)
            if not kml_name:
                raise HTTPException(status_code=400, detail="KMZ has no KML inside.")
            data = z.read(kml_name)
    k.from_string(data)

    def extract_polys(feat) -> List[Polygon | MultiPolygon]:
        out = []
        if hasattr(feat, "geometry") and feat.geometry is not None:
            try:
                g = shape(json.loads(feat.geometry.json))
                if isinstance(g, (Polygon, MultiPolygon)):
                    out.append(g)
            except Exception:
                pass
        if hasattr(feat, "features"):
            for f in feat.features():
                out.extend(extract_polys(f))
        return out

    geoms = []
    for d in k.features():          # Document(s)
        geoms.extend(extract_polys(d))

    mp = as_multipolygon(geoms)
    return mapping(mp)

@router.post("/upload")
async def upload_field(
    file: UploadFile = File(..., description="Shapefile ZIP (.zip), KML (.kml) or KMZ (.kmz)"),
    name: Optional[str] = Form(None)
):
    fname = (file.filename or "").lower()
    content = await file.read()

    try:
        if fname.endswith(".zip"):
            geom = shapefile_zip_to_geojson(content)
        elif fname.endswith(".kml"):
            geom = _kml_or_kmz_to_geojson(content, is_kmz=False)
        elif fname.endswith(".kmz"):
            geom = _kml_or_kmz_to_geojson(content, is_kmz=True)
        else:
            raise HTTPException(status_code=415, detail="Unsupported file type. Use .zip (shapefile), .kml, or .kmz.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Parse failed: {e}")

    # Area check
    try:
        area = area_ha(geom)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Area calculation failed: {e}")

    if area < MIN_FIELD_HA:
        raise HTTPException(status_code=400, detail=f"Field area {area:.2f} ha < minimum {MIN_FIELD_HA} ha")

    # Save to GCS
    field_id = uuid4().hex[:12]
    created_at = datetime.now(timezone.utc).isoformat()
    field_name = name or os.path.splitext(os.path.basename(fname))[0]

    meta = {
        "id": field_id,
        "name": field_name,
        "area_ha": round(area, 4),
        "created_at": created_at
    }
    geom_path = f"fields/{field_id}/field.geojson"
    meta_path = f"fields/{field_id}/meta.json"
    index_path = "fields/index.json"

    upload_json(geom, geom_path)
    upload_json(meta, meta_path)

    # Append to index
    try:
        idx = download_json(index_path) if exists(index_path) else []
        idx.append({"id": field_id, "name": field_name, "area_ha": round(area, 4), "created_at": created_at})
        upload_json(idx, index_path)
    except Exception:
        pass

    return {"ok": True, **meta}
