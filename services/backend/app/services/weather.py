"""Weather utilities: forecast fetching, aggregation, and precipitation tiles."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Mapping, Sequence

import requests

from app.utils.geometry import centroid_lonlat

OPEN_METEO_ENDPOINT = "https://api.open-meteo.com/v1/forecast"
RAINVIEWER_ENDPOINT = "https://api.rainviewer.com/public/weather-maps.json"
RAINVIEWER_TILE_TEMPLATE = "https://tilecache.rainviewer.com/v2/radar/{time}/256/{z}/{x}/{y}/2/1_1.png"
BASE_GDD_C = 10.0
SPRAY_MAX_WIND_KMH = 15.0
SPRAY_MAX_PRECIP_MM = 0.2

WMO_LOOKUP: dict[int, tuple[str, str]] = {
    0: ("Clear sky", "☀️"),
    1: ("Mainly clear", "🌤️"),
    2: ("Partly cloudy", "⛅️"),
    3: ("Overcast", "☁️"),
    45: ("Fog", "🌫️"),
    48: ("Depositing rime fog", "🌫️"),
    51: ("Light drizzle", "🌦️"),
    53: ("Moderate drizzle", "🌦️"),
    55: ("Dense drizzle", "🌧️"),
    61: ("Light rain", "🌧️"),
    63: ("Moderate rain", "🌧️"),
    65: ("Heavy rain", "🌧️"),
    66: ("Light freezing rain", "🌧️"),
    67: ("Heavy freezing rain", "🌧️"),
    71: ("Light snow", "❄️"),
    73: ("Moderate snow", "❄️"),
    75: ("Heavy snow", "❄️"),
    80: ("Rain showers", "🌦️"),
    81: ("Heavy showers", "🌧️"),
    82: ("Violent showers", "🌧️"),
    95: ("Thunderstorm", "⛈️"),
    96: ("Thunderstorm + hail", "⛈️"),
    99: ("Severe thunderstorm", "⛈️"),
}


class WeatherServiceError(RuntimeError):
    """Raised when an upstream weather provider fails."""


@dataclass
class DailyForecast:
    date: date
    temp_min_c: float
    temp_max_c: float
    precipitation_mm: float
    wind_avg_kmh: float
    wind_max_kmh: float
    description: str
    icon: str | None
    gdd: float
    spray_ok: bool


def _resolve_weather_code(code: int | None) -> tuple[str, str | None]:
    if code is None:
        return ("Forecast", None)
    return WMO_LOOKUP.get(code, ("Forecast", None))


def fetch_openmeteo_forecast(lat: float, lon: float) -> Mapping[str, Any]:
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max,weathercode",
        "timezone": "UTC",
        "windspeed_unit": "kmh",
    }
    try:
        response = requests.get(OPEN_METEO_ENDPOINT, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:  # pragma: no cover - network errors not deterministic
        raise WeatherServiceError(f"Open-Meteo request failed: {exc}") from exc


def fetch_rainviewer_tile() -> Dict[str, Any] | None:
    try:
        response = requests.get(RAINVIEWER_ENDPOINT, timeout=5)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:  # pragma: no cover
        raise WeatherServiceError(f"RainViewer request failed: {exc}") from exc

    radar = payload.get("radar", {}) or {}
    timeline: list[Mapping[str, Any]] = radar.get("nowcast") or radar.get("past") or []
    if not timeline:
        return None
    frame = timeline[-1]
    timestamp = frame.get("time")
    if timestamp is None:
        return None
    url = RAINVIEWER_TILE_TEMPLATE.format(time=timestamp)
    return {
        "token": f"rainviewer-{timestamp}",
        "urlTemplate": url,
        "minZoom": 0,
        "maxZoom": 12,
        "expiresAt": None,
    }


def _daily_entries_from_openmeteo(payload: Mapping[str, Any], base_temp_c: float) -> Sequence[DailyForecast]:
    daily = payload.get("daily") or {}
    times = daily.get("time") or []
    temps_max = daily.get("temperature_2m_max") or []
    temps_min = daily.get("temperature_2m_min") or []
    precipitation = daily.get("precipitation_sum") or []
    wind_max = daily.get("windspeed_10m_max") or []
    weather_codes = daily.get("weathercode") or []

    entries: list[DailyForecast] = []
    for idx, iso_date in enumerate(times):
        try:
            day = datetime.strptime(iso_date, "%Y-%m-%d").date()
        except ValueError:
            continue
        max_c = float(temps_max[idx]) if idx < len(temps_max) else 0.0
        min_c = float(temps_min[idx]) if idx < len(temps_min) else 0.0
        precip_mm = float(precipitation[idx]) if idx < len(precipitation) else 0.0
        wind_max_kmh = float(wind_max[idx]) if idx < len(wind_max) else 0.0
        description, icon = _resolve_weather_code(weather_codes[idx] if idx < len(weather_codes) else None)

        avg_temp = (max_c + min_c) / 2.0
        gdd = max(avg_temp - base_temp_c, 0.0)
        spray_ok = precip_mm <= SPRAY_MAX_PRECIP_MM and wind_max_kmh <= SPRAY_MAX_WIND_KMH
        entries.append(
            DailyForecast(
                date=day,
                temp_min_c=min_c,
                temp_max_c=max_c,
                precipitation_mm=round(precip_mm, 2),
                wind_avg_kmh=round(wind_max_kmh, 1),
                wind_max_kmh=round(wind_max_kmh, 1),
                description=description,
                icon=icon,
                gdd=round(gdd, 2),
                spray_ok=spray_ok,
            )
        )
    return entries


def build_weather_payload(aoi: Mapping[str, Any], *, base_temp_c: float = BASE_GDD_C) -> Dict[str, Any]:
    lat, lon = centroid_lonlat(aoi)
    forecast_raw = fetch_openmeteo_forecast(lat, lon)
    days = list(_daily_entries_from_openmeteo(forecast_raw, base_temp_c))[:5]
    if not days:
        raise WeatherServiceError("Weather data unavailable for AOI.")

    precip_tile = None
    try:
        precip_tile = fetch_rainviewer_tile()
    except WeatherServiceError:
        precip_tile = None

    cumulative_gdd = 0.0
    cumulative_precip = 0.0
    gdd_series: list[dict[str, Any]] = []
    precip_series: list[dict[str, Any]] = []
    for day in days:
        cumulative_gdd += day.gdd
        cumulative_precip += day.precipitation_mm
        gdd_series.append(
            {
                "date": day.date.isoformat(),
                "value": day.gdd,
                "cumulative": round(cumulative_gdd, 2),
            }
        )
        precip_series.append(
            {
                "date": day.date.isoformat(),
                "value": day.precipitation_mm,
                "cumulative": round(cumulative_precip, 2),
            }
        )

    spray_days = [day.date.isoformat() for day in days if day.spray_ok]

    return {
        "location": {"lat": lat, "lon": lon},
        "forecast": [
            {
                "date": day.date.isoformat(),
                "tempMinC": day.temp_min_c,
                "tempMaxC": day.temp_max_c,
                "precipitationMm": day.precipitation_mm,
                "windAvgKmh": day.wind_avg_kmh,
                "windMaxKmh": day.wind_max_kmh,
                "description": day.description,
                "icon": day.icon,
                "gdd": day.gdd,
                "sprayOk": day.spray_ok,
            }
            for day in days
        ],
        "gddBaseC": base_temp_c,
        "chart": {
            "gdd": gdd_series,
            "precipitation": precip_series,
        },
        "sprayRecommendation": {
            "bestDays": spray_days,
            "hasWindow": bool(spray_days),
        },
        "precipitationTile": precip_tile,
    }
