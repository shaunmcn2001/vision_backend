from fastapi import APIRouter, HTTPException
from app.services.tiles import (
    init_ee, load_field_geometry, ndvi_annual_image, ndvi_month_image, get_tile_template_for_image
)

router = APIRouter()

@router.get("/tiles/ndvi/annual/{field_id}/{year}")
def tiles_ndvi_annual(field_id: str, year: int):
    try:
        init_ee()
        geom = load_field_geometry(field_id)
        img = ndvi_annual_image(geom, year)
        t = get_tile_template_for_image(img)
        return {"ok": True, "field_id": field_id, "year": year, **t}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tile failed: {e}")

@router.get("/tiles/ndvi/month/{field_id}/{year}/{month}")
def tiles_ndvi_month(field_id: str, year: int, month: int):
    if month < 1 or month > 12:
        raise HTTPException(status_code=400, detail="month must be 1..12")
    try:
        init_ee()
        geom = load_field_geometry(field_id)
        img = ndvi_month_image(geom, year, month)
        t = get_tile_template_for_image(img)
        return {"ok": True, "field_id": field_id, "year": year, "month": month, **t}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tile failed: {e}")
