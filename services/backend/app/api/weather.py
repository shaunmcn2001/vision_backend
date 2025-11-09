"""Weather endpoints for forecast and spray planning."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.schemas import WeatherForecastRequest, WeatherForecastResponse
from app.services.weather import WeatherServiceError, build_weather_payload

router = APIRouter(prefix="/api", tags=["weather"])


@router.post("/weather/forecast", response_model=WeatherForecastResponse)
def weather_forecast(request: WeatherForecastRequest) -> WeatherForecastResponse:
    try:
        data = build_weather_payload(request.aoi, base_temp_c=request.base_temp_c)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except WeatherServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return WeatherForecastResponse(**data)
