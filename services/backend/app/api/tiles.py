from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Query

from app.services.indices import resolve_index
from app.services.tiles import (
    get_tile_template_for_image,
    index_annual_image,
    index_month_image,
    init_ee,
    load_field_geometry,
    resolve_clamp_range,
)

router = APIRouter()

def _collect_overrides(
    vis_min: float | None,
    vis_max: float | None,
    vis_palette: str | None,
) -> Dict[str, Any] | None:
    overrides: Dict[str, Any] = {}
    if vis_min is not None:
        overrides["min"] = vis_min
    if vis_max is not None:
        overrides["max"] = vis_max
    if vis_palette is not None:
        overrides["palette"] = vis_palette
    return overrides or None


@router.get("/tiles/ndvi/annual/{field_id}/{year}")
def tiles_ndvi_annual(
    field_id: str,
    year: int,
    index: str = Query("ndvi"),
    vis_min: float | None = Query(None),
    vis_max: float | None = Query(None),
    vis_palette: str | None = Query(None),
):
    overrides = _collect_overrides(vis_min, vis_max, vis_palette)

    try:
        init_ee()
        geom = load_field_geometry(field_id)
        definition, params = resolve_index(index)
        resolved_params = dict(params)
        img = index_annual_image(
            geom,
            year,
            definition=definition,
            parameters=resolved_params,
        )
        clamp_range = resolve_clamp_range(definition, overrides)
        if clamp_range is not None:
            img = img.clamp(*clamp_range)
        tile = get_tile_template_for_image(
            img,
            definition=definition,
            vis_overrides=overrides,
        )
        return {
            "ok": True,
            "field_id": field_id,
            "year": year,
            "index": definition.code,
            "band": definition.band_name,
            "parameters": resolved_params,
            "clamp_range": list(clamp_range) if clamp_range is not None else None,
            **tile,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tile failed: {e}")

@router.get("/tiles/ndvi/month/{field_id}/{year}/{month}")
def tiles_ndvi_month(
    field_id: str,
    year: int,
    month: int,
    index: str = Query("ndvi"),
    vis_min: float | None = Query(None),
    vis_max: float | None = Query(None),
    vis_palette: str | None = Query(None),
):
    if month < 1 or month > 12:
        raise HTTPException(status_code=400, detail="month must be 1..12")
    overrides = _collect_overrides(vis_min, vis_max, vis_palette)
    try:
        init_ee()
        geom = load_field_geometry(field_id)
        definition, params = resolve_index(index)
        resolved_params = dict(params)
        img = index_month_image(
            geom,
            year,
            month,
            definition=definition,
            parameters=resolved_params,
        )
        clamp_range = resolve_clamp_range(definition, overrides)
        if clamp_range is not None:
            img = img.clamp(*clamp_range)
        tile = get_tile_template_for_image(
            img,
            definition=definition,
            vis_overrides=overrides,
        )
        return {
            "ok": True,
            "field_id": field_id,
            "year": year,
            "month": month,
            "index": definition.code,
            "band": definition.band_name,
            "parameters": resolved_params,
            "clamp_range": list(clamp_range) if clamp_range is not None else None,
            **tile,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tile failed: {e}")
