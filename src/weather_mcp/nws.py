import httpx

from weather_mcp.errors import ServiceUnavailableError
from weather_mcp.http import get_with_retry
from weather_mcp.models import Coordinates


async def get_daily_forecast(
    client: httpx.AsyncClient, coordinates: Coordinates, days: int = 7
) -> list[str]:
    """Fetches the forecast for the given coordinates from the NWS API."""
    point_data = await get_point_data(client, coordinates)
    forecast_url = point_data.get("properties", {}).get("forecast")
    if not forecast_url:
        raise ServiceUnavailableError("Forecast URL not found in point data.")
    forecast_data = await get_forecast_data(client, forecast_url)
    forecast_count = min(days * 2, len(forecast_data))
    forecast_data = forecast_data[:forecast_count]
    data = [
        f"{period['name']}: {period['temperature']}{period['temperatureUnit']} - {period['shortForecast']}"
        for period in forecast_data
    ]
    return data


async def get_hourly_forecast(
    client: httpx.AsyncClient, coordinates: Coordinates, hours: int = 24
) -> list[str]:
    """Fetches the hourly forecast for the given coordinates from the NWS API."""
    point_data = await get_point_data(client, coordinates)
    forecast_url = point_data.get("properties", {}).get("forecastHourly")
    if not forecast_url:
        raise ServiceUnavailableError("Forecast URL not found in point data.")
    forecast_data = await get_forecast_data(client, forecast_url)
    forecast_count = min(hours, len(forecast_data))
    forecast_data = forecast_data[:forecast_count]
    data = [
        f"{period['startTime']}: {period['temperature']}{period['temperatureUnit']} - {period['shortForecast']}"
        for period in forecast_data
    ]
    return data


async def get_point_data(client: httpx.AsyncClient, coordinates: Coordinates) -> dict:
    """Fetches the point forecast for the given coordinates from the NWS API."""
    url = f"https://api.weather.gov/points/{coordinates.lat},{coordinates.lon}"
    response = await get_weather_data(client, url)
    return response


async def get_active_alerts(
    client: httpx.AsyncClient, coordinates: Coordinates
) -> list[str]:
    """Fetches the active alerts for the given coordinates from the NWS API."""
    url = "https://api.weather.gov/alerts/active"
    params = {"point": f"{coordinates.lat},{coordinates.lon}"}
    response = await get_weather_data(client, url, params=params)
    features = response.get("features", [])
    if len(features) == 0:
        return ["No active alerts for this location."]
    alerts = [
        f"{feature['properties']['severity']} - {feature['properties']['event']} - {feature['properties']['headline']} - {feature['properties']['description']}"
        for feature in features
    ]
    return alerts


async def get_weather_discussion(
    client: httpx.AsyncClient, coordinates: Coordinates, count: int = 1
) -> list[str]:
    """Fetches the weather discussions for the given gridId from the NWS API."""
    point_data = await get_point_data(client, coordinates)
    grid_id = point_data.get("properties", {}).get("gridId")
    if not grid_id:
        raise ServiceUnavailableError("gridId not found in point data.")
    url = f"https://api.weather.gov/products/types/AFD/locations/{grid_id}"
    discussions = await get_weather_data(client, url)
    discussions = discussions.get("@graph", [])
    discussions_count = min(count, len(discussions))
    if discussions_count == 0:
        return [f"No discussions found for {url}"]
    discussions = discussions[:discussions_count]
    data = []
    for discussion in discussions:
        dis_data = await get_weather_data(client, discussion["@id"])
        data.append(dis_data["productText"])
    return data


async def get_current_conditions(
    client: httpx.AsyncClient, coordinates: Coordinates
) -> str:
    """Fetches the current conditions for the given coordinates from the NWS API."""
    point_data = await get_point_data(client, coordinates)
    observation_stations_url = point_data.get("properties", {}).get(
        "observationStations"
    )
    if not observation_stations_url:
        raise ServiceUnavailableError(
            "Observation stations URL not found in point data."
        )
    stations = await get_weather_data(client, observation_stations_url)
    for station in stations["features"]:
        weather = await get_station_weather(client, station["id"])
        if weather["is_valid"]:
            return f"The current weather at {weather['station_name']} is {weather['temperature']} and {weather['description']}"
    return "None of the weather stations are reporting valid data"


async def get_forecast_data(client: httpx.AsyncClient, url: str) -> list[dict]:
    """Fetches the forecast data from the given URL."""
    response = await get_weather_data(client, url)
    return response.get("properties", {}).get("periods", [])


async def get_station_weather(client: httpx.AsyncClient, url: str) -> dict:
    url = f"{url}/observations/latest"
    UNIT_CODES = {
        "wmoUnit:degC": "C",
        "wmoUnit:degF": "F",
    }
    weather_data = await get_weather_data(client, url)
    weather = {}
    weather["station_name"] = weather_data["properties"]["stationName"]
    weather["is_valid"] = (
        weather_data.get("properties", {})
        .get("temperature", {})
        .get("qualityControl", "")
        .lower()
        == "v"
        and len(weather_data.get("properties", {}).get("textDescription", "")) > 0
        and weather_data.get("properties", {}).get("temperature", {}).get("unitCode")
        in UNIT_CODES
    )
    if weather["is_valid"]:
        weather["temperature"] = (
            f"{weather_data['properties']['temperature']['value']}°{UNIT_CODES[weather_data['properties']['temperature']['unitCode']]}"
        )
        weather["description"] = weather_data["properties"]["textDescription"]
    return weather


async def get_weather_data(client: httpx.AsyncClient, url: str, **kwargs) -> dict:
    """Fetches the weather data for the given coordinates from the NWS API."""
    response = await get_with_retry(client, url, **kwargs)
    return response
