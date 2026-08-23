import pytest

from weather_mcp import nws as nws_mod
from weather_mcp.errors import (
    ServiceUnavailableError,
)
from weather_mcp.models import Coordinates

POINT_RESPONSE = {
    "properties": {
        "forecast": "https://api.weather.gov/gridpoints/RAH/32,66/forecast",
        "forecastHourly": "https://api.weather.gov/gridpoints/RAH/32,66/forecast/hourly",
    }
}
DAILY_FORECAST_RESPONSE = {
    "properties": {
        "periods": [
            {
                "number": 1,
                "name": "Today",
                "startTime": "2026-08-20T19:00:00-04:00",
                "temperature": 70,
                "temperatureUnit": "F",
                "windSpeed": "6 mph",
                "windDirection": "SW",
                "icon": "https://api.weather.gov/icons/land/night/sct?size=small",
                "shortForecast": "Chance Showers And Thunderstorms",
            },
            {
                "number": 2,
                "name": "Tonight",
                "startTime": "2026-08-20T20:00:00-04:00",
                "temperature": 83,
                "temperatureUnit": "F",
                "windSpeed": "3 mph",
                "windDirection": "SW",
                "icon": "https://api.weather.gov/icons/land/night/tsra_hi,20?size=small",
                "shortForecast": "Slight Chance Showers And Thunderstorms",
            },
        ]
    }
}

HOURLY_FORECAST_RESPONSE = {
    "properties": {
        "periods": [
            {
                "number": 1,
                "startTime": "2026-08-20T19:00:00-04:00",
                "temperature": 86,
                "temperatureUnit": "F",
                "windSpeed": "6 mph",
                "windDirection": "SW",
                "icon": "https://api.weather.gov/icons/land/night/sct?size=small",
                "shortForecast": "Partly Cloudy",
            },
            {
                "number": 2,
                "startTime": "2026-08-20T20:00:00-04:00",
                "temperature": 83,
                "temperatureUnit": "F",
                "windSpeed": "3 mph",
                "windDirection": "SW",
                "icon": "https://api.weather.gov/icons/land/night/tsra_hi,20?size=small",
                "shortForecast": "Slight Chance Showers And Thunderstorms",
            },
        ]
    }
}


@pytest.mark.asyncio
async def test_get_point_info_success(monkeypatch):
    async def fake_get_with_retry(client, url, **kwargs):
        return POINT_RESPONSE

    monkeypatch.setattr("weather_mcp.nws.get_with_retry", fake_get_with_retry)

    point_info = await nws_mod.get_point_data(
        client=None, coordinates=Coordinates(lat=36.0956, lon=-79.8269)
    )

    assert (
        point_info["properties"]["forecast"] == POINT_RESPONSE["properties"]["forecast"]
    )
    assert (
        point_info["properties"]["forecastHourly"]
        == POINT_RESPONSE["properties"]["forecastHourly"]
    )


@pytest.mark.asyncio
async def test_daily_data_success(monkeypatch):
    async def fake_get_with_retry(client, url, **kwargs):
        return DAILY_FORECAST_RESPONSE

    monkeypatch.setattr("weather_mcp.nws.get_with_retry", fake_get_with_retry)

    forecast = await nws_mod.get_forecast_data(
        client=None, url="https://api.weather.gov/gridpoints/RAH/32,66/forecast"
    )

    assert (
        forecast[0]["shortForecast"]
        == DAILY_FORECAST_RESPONSE["properties"]["periods"][0]["shortForecast"]
    )


@pytest.mark.asyncio
async def test_daily_forecast_missing_forecast_url(monkeypatch):
    async def fake_get_with_retry(client, url, **kwargs):
        return {}

    monkeypatch.setattr("weather_mcp.nws.get_with_retry", fake_get_with_retry)

    with pytest.raises(ServiceUnavailableError) as exc_info:
        await nws_mod.get_daily_forecast(
            client=None, coordinates=Coordinates(lat=36.0956, lon=-79.8269)
        )

    assert str(exc_info.value) == "Forecast URL not found in point data."


@pytest.mark.asyncio
async def test_hourly_data_success(monkeypatch):
    async def fake_get_with_retry(client, url, **kwargs):
        return HOURLY_FORECAST_RESPONSE

    monkeypatch.setattr("weather_mcp.nws.get_with_retry", fake_get_with_retry)

    forecast = await nws_mod.get_forecast_data(
        client=None, url="https://api.weather.gov/gridpoints/RAH/32,66/forecast/hourly"
    )

    assert (
        forecast[0]["shortForecast"]
        == HOURLY_FORECAST_RESPONSE["properties"]["periods"][0]["shortForecast"]
    )


@pytest.mark.asyncio
async def test_hourly_forecast_missing_forecast_url(monkeypatch):
    async def fake_get_with_retry(client, url, **kwargs):
        return {}

    monkeypatch.setattr("weather_mcp.nws.get_with_retry", fake_get_with_retry)

    with pytest.raises(ServiceUnavailableError) as exc_info:
        await nws_mod.get_hourly_forecast(
            client=None, coordinates=Coordinates(lat=36.0956, lon=-79.8269)
        )

    assert str(exc_info.value) == "Forecast URL not found in point data."


@pytest.mark.asyncio
async def test_get_point_data_uses_points_endpoint(monkeypatch):
    """Regression test: get_point_data previously pointed at /active/ instead of /points/."""
    captured = {}

    async def fake_get_with_retry(client, url, **kwargs):
        captured["url"] = url
        return POINT_RESPONSE

    monkeypatch.setattr("weather_mcp.nws.get_with_retry", fake_get_with_retry)

    await nws_mod.get_point_data(
        client=None, coordinates=Coordinates(lat=36.0956, lon=-79.8269)
    )

    assert captured["url"] == "https://api.weather.gov/points/36.0956,-79.8269"


@pytest.mark.asyncio
async def test_active_alerts_with_alerts(monkeypatch):
    ALERTS_RESPONSE = {
        "features": [
            {
                "properties": {
                    "severity": "Moderate",
                    "event": "Heat Advisory",
                    "headline": "Heat Advisory issued",
                    "description": "Hot and humid conditions expected.",
                }
            }
        ]
    }

    async def fake_get_with_retry(client, url, **kwargs):
        return ALERTS_RESPONSE

    monkeypatch.setattr("weather_mcp.nws.get_with_retry", fake_get_with_retry)

    alerts = await nws_mod.get_active_alerts(
        client=None, coordinates=Coordinates(lat=36.0956, lon=-79.8269)
    )

    assert alerts == [
        "Moderate - Heat Advisory - Heat Advisory issued - Hot and humid conditions expected."
    ]


@pytest.mark.asyncio
async def test_active_alerts_none(monkeypatch):
    async def fake_get_with_retry(client, url, **kwargs):
        return {"features": []}

    monkeypatch.setattr("weather_mcp.nws.get_with_retry", fake_get_with_retry)

    alerts = await nws_mod.get_active_alerts(
        client=None, coordinates=Coordinates(lat=36.0956, lon=-79.8269)
    )

    assert alerts == ["No active alerts for this location."]


@pytest.mark.asyncio
async def test_station_weather_valid_celsius(monkeypatch):
    async def fake_get_with_retry(client, url, **kwargs):
        return {
            "properties": {
                "stationName": "Test Station",
                "textDescription": "Clear",
                "temperature": {
                    "qualityControl": "V",
                    "value": 20,
                    "unitCode": "wmoUnit:degC",
                },
            }
        }

    monkeypatch.setattr("weather_mcp.nws.get_with_retry", fake_get_with_retry)

    weather = await nws_mod.get_station_weather(
        client=None, url="https://api.weather.gov/stations/TEST"
    )

    assert weather["is_valid"] is True
    assert weather["temperature"] == "20°C"
    assert weather["description"] == "Clear"


@pytest.mark.asyncio
async def test_station_weather_valid_fahrenheit(monkeypatch):
    async def fake_get_with_retry(client, url, **kwargs):
        return {
            "properties": {
                "stationName": "Test Station",
                "textDescription": "Clear",
                "temperature": {
                    "qualityControl": "V",
                    "value": 68,
                    "unitCode": "wmoUnit:degF",
                },
            }
        }

    monkeypatch.setattr("weather_mcp.nws.get_with_retry", fake_get_with_retry)

    weather = await nws_mod.get_station_weather(
        client=None, url="https://api.weather.gov/stations/TEST"
    )

    assert weather["is_valid"] is True
    assert weather["temperature"] == "68°F"


@pytest.mark.asyncio
async def test_station_weather_unrecognized_unit(monkeypatch):
    async def fake_get_with_retry(client, url, **kwargs):
        return {
            "properties": {
                "stationName": "Test Station",
                "textDescription": "Clear",
                "temperature": {
                    "qualityControl": "V",
                    "value": 293,
                    "unitCode": "wmoUnit:K",
                },
            }
        }

    monkeypatch.setattr("weather_mcp.nws.get_with_retry", fake_get_with_retry)

    weather = await nws_mod.get_station_weather(
        client=None, url="https://api.weather.gov/stations/TEST"
    )

    assert weather["is_valid"] is False
    assert "temperature" not in weather


@pytest.mark.asyncio
async def test_station_weather_missing_unit_code(monkeypatch):
    async def fake_get_with_retry(client, url, **kwargs):
        return {
            "properties": {
                "stationName": "Test Station",
                "textDescription": "Clear",
                "temperature": {"qualityControl": "V", "value": 20},
            }
        }

    monkeypatch.setattr("weather_mcp.nws.get_with_retry", fake_get_with_retry)

    weather = await nws_mod.get_station_weather(
        client=None, url="https://api.weather.gov/stations/TEST"
    )

    assert weather["is_valid"] is False
    assert "temperature" not in weather


@pytest.mark.asyncio
async def test_current_conditions_skips_invalid_station(monkeypatch):
    POINT = {
        "properties": {
            "observationStations": "https://api.weather.gov/gridpoints/RAH/32,66/stations"
        }
    }
    STATIONS = {
        "features": [
            {"id": "https://api.weather.gov/stations/BAD"},
            {"id": "https://api.weather.gov/stations/GOOD"},
        ]
    }
    BAD_OBS = {
        "properties": {
            "stationName": "Bad Station",
            "textDescription": "",
            "temperature": {
                "qualityControl": "S",
                "value": 10,
                "unitCode": "wmoUnit:degC",
            },
        }
    }
    GOOD_OBS = {
        "properties": {
            "stationName": "Good Station",
            "textDescription": "Sunny",
            "temperature": {
                "qualityControl": "V",
                "value": 25,
                "unitCode": "wmoUnit:degC",
            },
        }
    }

    async def fake_get_with_retry(client, url, **kwargs):
        if "/points/" in url:
            return POINT
        if url.endswith("/stations"):
            return STATIONS
        if "BAD" in url:
            return BAD_OBS
        return GOOD_OBS

    monkeypatch.setattr("weather_mcp.nws.get_with_retry", fake_get_with_retry)

    result = await nws_mod.get_current_conditions(
        client=None, coordinates=Coordinates(lat=36.0956, lon=-79.8269)
    )

    assert result == "The current weather at Good Station is 25°C and Sunny"


@pytest.mark.asyncio
async def test_current_conditions_no_valid_stations(monkeypatch):
    POINT = {
        "properties": {
            "observationStations": "https://api.weather.gov/gridpoints/RAH/32,66/stations"
        }
    }
    STATIONS = {"features": [{"id": "https://api.weather.gov/stations/BAD"}]}
    BAD_OBS = {
        "properties": {
            "stationName": "Bad Station",
            "textDescription": "",
            "temperature": {
                "qualityControl": "S",
                "value": 10,
                "unitCode": "wmoUnit:degC",
            },
        }
    }

    async def fake_get_with_retry(client, url, **kwargs):
        if "/points/" in url:
            return POINT
        if url.endswith("/stations"):
            return STATIONS
        return BAD_OBS

    monkeypatch.setattr("weather_mcp.nws.get_with_retry", fake_get_with_retry)

    result = await nws_mod.get_current_conditions(
        client=None, coordinates=Coordinates(lat=36.0956, lon=-79.8269)
    )

    assert result == "None of the weather stations are reporting valid data"


@pytest.mark.asyncio
async def test_current_conditions_missing_observation_stations(monkeypatch):
    async def fake_get_with_retry(client, url, **kwargs):
        return {"properties": {}}

    monkeypatch.setattr("weather_mcp.nws.get_with_retry", fake_get_with_retry)

    with pytest.raises(ServiceUnavailableError) as exc_info:
        await nws_mod.get_current_conditions(
            client=None, coordinates=Coordinates(lat=36.0956, lon=-79.8269)
        )

    assert str(exc_info.value) == "Observation stations URL not found in point data."


@pytest.mark.asyncio
async def test_weather_discussion_success(monkeypatch):
    POINT = {"properties": {"gridId": "RAH"}}
    LOCATIONS = {"@graph": [{"@id": "https://api.weather.gov/products/abc123"}]}
    PRODUCT = {"productText": "Area Forecast Discussion text."}

    async def fake_get_with_retry(client, url, **kwargs):
        if "/points/" in url:
            return POINT
        if "/locations/" in url:
            return LOCATIONS
        return PRODUCT

    monkeypatch.setattr("weather_mcp.nws.get_with_retry", fake_get_with_retry)

    discussions = await nws_mod.get_weather_discussion(
        client=None, coordinates=Coordinates(lat=36.0956, lon=-79.8269)
    )

    assert discussions == ["Area Forecast Discussion text."]


@pytest.mark.asyncio
async def test_weather_discussion_none_found(monkeypatch):
    POINT = {"properties": {"gridId": "RAH"}}

    async def fake_get_with_retry(client, url, **kwargs):
        if "/points/" in url:
            return POINT
        return {"@graph": []}

    monkeypatch.setattr("weather_mcp.nws.get_with_retry", fake_get_with_retry)

    discussions = await nws_mod.get_weather_discussion(
        client=None, coordinates=Coordinates(lat=36.0956, lon=-79.8269)
    )

    assert discussions == [
        "No discussions found for https://api.weather.gov/products/types/AFD/locations/RAH"
    ]


@pytest.mark.asyncio
async def test_weather_discussion_missing_grid_id(monkeypatch):
    async def fake_get_with_retry(client, url, **kwargs):
        return {"properties": {}}

    monkeypatch.setattr("weather_mcp.nws.get_with_retry", fake_get_with_retry)

    with pytest.raises(ServiceUnavailableError) as exc_info:
        await nws_mod.get_weather_discussion(
            client=None, coordinates=Coordinates(lat=36.0956, lon=-79.8269)
        )

    assert str(exc_info.value) == "gridId not found in point data."
