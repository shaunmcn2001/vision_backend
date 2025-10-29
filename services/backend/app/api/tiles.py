from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response

from ee import deserializer

from app.models.schemas import TileSessionRequest, TileSessionResponse
from app.services.earth_engine import ensure_ee
from app.services.tiles import create_tile_session, fetch_tile_bytes

router = APIRouter(prefix="/api", tags=["tiles"])


@router.post("/tiles/session", response_model=TileSessionResponse)
def create_tiles_session(payload: TileSessionRequest) -> TileSessionResponse:
    ensure_ee()
    try:
        image = deserializer.fromJSON(payload.image)
    except Exception as exc:  # pragma: no cover - ee raises custom errors
        raise HTTPException(status_code=400, detail=f"Invalid image specification: {exc}")

    info = create_tile_session(
        image,
        vis_params=payload.vis_params,
        min_zoom=payload.min_zoom,
        max_zoom=payload.max_zoom,
    )
    return TileSessionResponse(**info)


@router.get("/tiles/{token}/{z}/{x}/{y}")
def proxy_tile(token: str, z: int, x: int, y: int) -> Response:
    try:
        data, content_type = fetch_tile_bytes(token, z, x, y)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Tile proxy failed: {exc}") from exc
    return Response(content=data, media_type=content_type)
