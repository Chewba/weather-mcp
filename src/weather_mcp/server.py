import asyncio
import logging

import httpx
from mcp.server.mcpserver import MCPServer

mcp = MCPServer(name="weather", instructions="...")

from weather_mcp import nws
from weather_mcp.config import USER_AGENT
from weather_mcp.errors import WeatherMcpError
from weather_mcp.geocode import get_coordinates
from weather_mcp.rag.ingest import ingest_discussion

logger = logging.getLogger(__name__)


@mcp.tool()
async def get_daily_forecast(address: str, days: int = 3) -> str:
    """Fetches a multi-day weather forecast (USA locations only) for a plain text city, and state or zip code. Use this for a multi-day outlook (e.g. "the next few days" or "this weekend") -- not for the current conditions or a specific hour. Returns one summary per day/night period, defaulting to the next 3 days (optional days param)."""
    async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}) as client:
        response = await _get_daily_forecast(client, address, days)
        return response


@mcp.tool()
async def get_current_conditions(address: str) -> str:
    """Fetches the current weather conditions right now (USA locations only) for a plain text city, and state or zip code. Use this for "what's the weather like right now" -- not for a forecast. Returns a plain text description of the current conditions."""
    async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}) as client:
        try:
            coordinates = await get_coordinates(client, address)
            conditions = await nws.get_current_conditions(client, coordinates)
            return conditions
        except WeatherMcpError as e:
            return str(e)


@mcp.tool()
async def get_active_alerts(address: str) -> str:
    """Fetches active weather alerts (USA locations only) for a plain text city, and state or zip code, and returns a plain text description of the active alerts."""
    async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}) as client:
        try:
            coordinates = await get_coordinates(client, address)
            alerts = await nws.get_active_alerts(client, coordinates)
            return "\n".join(alerts)
        except WeatherMcpError as e:
            return str(e)


@mcp.tool()
async def get_weather_discussion(address: str, count: int = 1) -> str:
    """Fetches the National Weather Service forecaster's technical discussion (USA locations only) for a plain text city, and state or zip code -- the meteorologist's own narrative reasoning behind the forecast. Only use this when the user explicitly asks for the forecaster's discussion or reasoning, not for a normal forecast request. Optional count param controls how many discussions to return, defaulting to 1. Returns the plain text of the requested discussions."""
    async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}) as client:
        try:
            coordinates = await get_coordinates(client, address)
            discussions = await nws.get_weather_discussion(client, coordinates, count)
        except WeatherMcpError as e:
            return str(e)
        if not discussions:
            return f"No discussions found for {address}"
        for discussion in discussions:
            try:
                await ingest_discussion(discussion, discussion["issuingOffice"])
            except Exception:
                logger.exception("Failed to ingest weather discussion %s", discussion.get("@id"))
        return "\n".join(discussion["productText"] for discussion in discussions)


@mcp.tool()
async def get_hourly_forecast(address: str, hours: int = 24) -> str:
    """Fetches an hour-by-hour weather forecast (USA locations only) for a plain text city, and state or zip code. Use this for a specific near-term time (e.g. "in an hour" or "tonight at 9pm") that needs hour-level detail -- not for a multi-day outlook. Optional hours param controls how many hours out to return, defaulting to 24."""
    async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}) as client:
        try:
            coordinates = await get_coordinates(client, address)
            forecast = await nws.get_hourly_forecast(client, coordinates, hours)
            return "\n".join(forecast)
        except WeatherMcpError as e:
            return str(e)


@mcp.tool()
async def compare_forecasts(address1: str, address2: str, days: int = 3) -> str:
    """Compares the daily weather forecasts for two locations (USA only), given as city, and state or zip code. This already fetches the daily forecast for both addresses internally -- do not also call get_daily_forecast separately for either address. Returns address1's forecast with a newline between periods, then address2's."""
    async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}) as client:
        response = await asyncio.gather(
            _get_daily_forecast(client, address1, days),
            _get_daily_forecast(client, address2, days),
        )
        return f"Forecast for {address1}:\n{response[0]}\n\nForecast for {address2}:\n{response[1]}"


async def _get_daily_forecast(
    client: httpx.AsyncClient, address: str, days: int = 3
) -> str:
    """Fetches the weather forecast for a united states plain text city, and state or zip code, you can send in an optional days (default 3), and returns a plain text forecast for the next 3 days (day/night)."""
    try:
        coordinates = await get_coordinates(client, address)
        forecast = await nws.get_daily_forecast(client, coordinates, days)
        return "\n".join(forecast)
    except WeatherMcpError as e:
        return str(e)


def main() -> None:
    """Main entry point for the weather-mcp server."""
    mcp.run()
