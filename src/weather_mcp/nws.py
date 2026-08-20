import httpx
from weather_mcp.models import Coordinates
from weather_mcp.errors import ServiceUnavailableError
from weather_mcp.http import get_with_retry

async def get_daily_forecast(client: httpx.AsyncClient, coordinates: Coordinates, days: int = 7) -> list[str]:
    """Fetches the forecast for the given coordinates from the NWS API."""
    point_data = await get_point_data(client, coordinates)
    forecast_url = point_data.get("properties", {}).get("forecast")
    if not forecast_url:
        raise ServiceUnavailableError("Forecast URL not found in point data.")
    forecast_data = await get_forecast_data(client, forecast_url)
    forecast_count = min(days * 2, len(forecast_data))
    forecast_data = forecast_data[:forecast_count]
    data = [f"{period['name']}: {period['temperature']}{period['temperatureUnit']} - {period['shortForecast']}" for period in forecast_data]
    return data

async def get_hourly_forecast(client: httpx.AsyncClient, coordinates: Coordinates, hours: int = 24) -> list[str]:
    """Fetches the hourly forecast for the given coordinates from the NWS API."""
    point_data = await get_point_data(client, coordinates)
    forecast_url = point_data.get("properties", {}).get("forecastHourly")
    if not forecast_url:
        raise ServiceUnavailableError("Forecast URL not found in point data.")
    forecast_data = await get_forecast_data(client, forecast_url)
    forecast_count = min(hours, len(forecast_data))
    forecast_data = forecast_data[:forecast_count]
    data = [f"{period['startTime']}: {period['temperature']}{period['temperatureUnit']} - {period['shortForecast']}" for period in forecast_data]
    return data


async def get_point_data(client: httpx.AsyncClient, coordinates: Coordinates) -> dict:
    """Fetches the point forecast for the given coordinates from the NWS API."""
    url = f"https://api.weather.gov/points/{coordinates.lat},{coordinates.lon}"
    response = await get_weather_data(client, url)
    return response

async def get_forecast_data(client: httpx.AsyncClient, url: str) -> list[dict]:
    """Fetches the forecast data from the given URL."""
    response = await get_weather_data(client, url)
    return response.get("properties", {}).get("periods", [])

async def get_weather_data(client: httpx.AsyncClient, url: str, **kwargs) -> dict:
    """Fetches the weather data for the given coordinates from the NWS API."""
    response = await get_with_retry(client, url, **kwargs)
    return response