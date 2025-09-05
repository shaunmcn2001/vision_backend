import os
from uuid import uuid4
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from shapely.geometry import shape, mapping
from shapely.validation import make_valid
from shapely.ops import transform as shp_transform
from pyproj import Transformer
from app.services.gcs import upload_json, download_json, exists

router = APIRouter()
MIN_FIELD_HA = float(os.getenv("MIN_FIELD_HA", "1.0"))  # default 1 ha

# ---- helpers ----
_transformer = Transformer.from_crs("EPSG:4326", "EPSG:3577", always_xy=True)  # Australia Albers (equal-area)

def _area_m2(geojson_geom: dict) -> float:
    geom = make_valid(shape(geojson_geom))
    reproj = shp_transform(lambda x, y, z=None: _transformer.transform(x, y), geom)
    return float(reproj.area)

def _validate_geometry(geom: dict):
    t = geom.get("type")
    if t not in ("Polygon", "MultiPolygon"):
        raise HTTPException(status_code=400, detail=f"geometry.type must be Polygon or MultiPolygon, got {t}")
    # Additional checks (self-intersections fixed by make_valid in area calc)

# ---- models ----
class FieldCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    geometry: dict = Field(..., description="GeoJSON Polygon/MultiPolygon in EPSG:4326")

class FieldSummary(BaseModel):
    id: str
    name: str
    area_ha: float
    created_at: str

class FieldDetail(FieldSummary):
    geometry: dict

# ---- endpoints ----
@router.post("", response_model=FieldDetail)
def create_field(payload: FieldCreate):
    _validate_geometry(payload.geometry)
    area_m2 = _area_m2(payload.geometry)
    area_ha = area_m2 / 10000.0
    if area_ha < MIN_FIELD_HA:
        raise HTTPException(
            status_code=400,
            detail=f"Field area {area_ha:.2f} ha is smaller than minimum {MIN_FIELD_HA} ha"
        )

    field_id = uuid4().hex[:12]
    created_at = datetime.now(timezone.utc).isoformat()

    # save geometry and metadata to GCS
    meta = {
        "id": field_id,
        "name": payload.name,
        "area_ha": round(area_ha, 4),
        "created_at": created_at,
    }
    geom_path = f"fields/{field_id}/field.geojson"
    meta_path = f"fields/{field_id}/meta.json"
    index_path = "fields/index.json"

    upload_json(payload.geometry, geom_path)
    upload_json(meta, meta_path)

    # update index.json (best-effort; if missing, create)
    try:
        idx = download_json(index_path) if exists(index_path) else []
        # keep small summary only
        idx.append({"id": field_id, "name": payload.name, "area_ha": round(area_ha, 4), "created_at": created_at})
        upload_json(idx, index_path)
    except Exception:
        # Non-fatal if index update fails
        pass

    return {**meta, "geometry": payload.geometry}

@router.get("", response_model=list[FieldSummary])
def list_fields():
    index_path = "fields/index.json"
    if not exists(index_path):
        return []
    return download_json(index_path)

@router.get("/{field_id}", response_model=FieldDetail)
def get_field(field_id: str):
    meta_path = f"fields/{field_id}/meta.json"
    geom_path = f"fields/{field_id}/field.geojson"
    if not exists(meta_path) or not exists(geom_path):
        raise HTTPException(status_code=404, detail="Field not found")
    meta = download_json(meta_path)
    geom = download_json(geom_path)
    return {**meta, "geometry": geom}
