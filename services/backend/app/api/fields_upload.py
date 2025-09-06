from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional, List
import os, io, zipfile, json, tempfile
import shapefile  # pyshp
from fastkml import kml
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid
from pyproj import Transformer
from app.services.gcs import upload_json, download_json, exists
from uuid import uuid4
from datetime import datetime, timezone

router = APIRouter()

MIN_FIELD_HA = float(os.getenv("MIN_FIELD_HA", "1.0"))  # default 1 ha
# Equal-area for AU
_tx = Transformer.from_crs("EPSG:4326", "EPSG:3577", always_xy=True)

def _area_ha(geom_geojson: dict) -> float:
    g = make_valid(shape(geom_geojson))
    # reproject to equal-area
    g2 = shapely.ops.transform(lambda x, y, z=None: _tx.transform(x, y), g)
    return float(g2.area) / 10000.0

def _as_multipolygon(geoms: List[Polygon | MultiPolygon]):
    polys = []
    for g in geoms:
        if isinstance(g, Polygon):
            polys.append(g)
        elif isinstance(g, MultiPolygon):
            polys.extend(list(g.geoms))
    if not polys:
        raise HTTPException(status_code=400, detail="No polygon features found.")
    merged = unary_union(polys)
    if isinstance(merged, Polygon):
        return MultiPolygon([merged])
    if isinstance(merged, MultiPolygon):
        return merged
    raise HTTPException(status_code=400, detail="Could not merge polygons.")

def _shapefile_zip_to_geojson(file_bytes: bytes) -> dict:
    """
    Accept a ZIP that contains .shp, .dbf, .shx (and optionally .prj).
    Returns a GeoJSON MultiPolygon in EPSG:4326.
    """
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(io.BytesIO(file_bytes), "r") as z:
            z.extractall(td)
        # find .shp
        shp_path = None
        for n in os.listdir(td):
            if n.lower().endswith(".shp"):
                shp_path = os.path.join(td, n)
                break
        if not shp_path:
            raise HTTPException(status_code=400, detail="ZIP must contain a .shp file.")

        r = shapefile.Reader(shp_path)
        geoms = []
        for s in r.shapes():
            try:
                geom = shape(s.__geo_interface__)
                geoms.append(geom)
            except Exception:
                continue
        mp = _as_multipolygon(geoms)
        return mapping(mp)  # EPSG:4326 assumed

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

    mp = _as_multipolygon(geoms)
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
            geom = _shapefile_zip_to_geojson(content)
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
        area = _area_ha(geom)
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
