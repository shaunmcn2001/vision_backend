import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app  # noqa: E402


def test_weather_forecast_endpoint(monkeypatch):
    from app.services import weather as weather_service

    monkeypatch.setattr(weather_service, "centroid_lonlat", lambda _aoi: (-35.0, 149.0))

    sample_openmeteo = {
        "daily": {
            "time": ["2025-05-01", "2025-05-02"],
            "temperature_2m_max": [18.0, 17.0],
            "temperature_2m_min": [8.0, 7.0],
            "precipitation_sum": [0.0, 1.4],
            "windspeed_10m_max": [10.0, 12.0],
            "weathercode": [0, 63],
        }
    }

    sample_rainviewer = {
        "radar": {
            "past": [
                {"time": 111},
                {"time": 222},
            ]
        }
    }

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

        def raise_for_status(self):
            return None

    def fake_get(url, params=None, **kwargs):  # pylint: disable=unused-argument
        if "open-meteo" in url:
            return FakeResponse(sample_openmeteo)
        if "rainviewer" in url:
            return FakeResponse(sample_rainviewer)
        raise AssertionError(f"Unexpected URL {url}")

    monkeypatch.setattr(weather_service.requests, "get", fake_get)

    client = TestClient(app)
    payload = {
        "aoi": {
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
        }
    }
    response = client.post("/api/weather/forecast", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert len(body["forecast"]) == 2
    assert body["forecast"][0]["icon"] == "☀️"
    assert body["sprayRecommendation"]["hasWindow"] is True
    assert body["precipitationTile"]["urlTemplate"].startswith("https://tilecache.rainviewer.com")


def test_weather_provider_error(monkeypatch):
    from app.services import weather as weather_service

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

        def raise_for_status(self):
            return None

    def fake_get(url, params=None, **kwargs):  # pylint: disable=unused-argument
        if "open-meteo" in url:
            return FakeResponse({"daily": {}})
        if "rainviewer" in url:
            return FakeResponse({})
        raise AssertionError(f"Unexpected URL {url}")

    monkeypatch.setattr(weather_service.requests, "get", fake_get)

    client = TestClient(app)
    payload = {
        "aoi": {
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
        }
    }
    response = client.post("/api/weather/forecast", json=payload)
    assert response.status_code == 503
