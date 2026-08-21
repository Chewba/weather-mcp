
import httpx
from mcp.server.mcpserver import MCPServer

mcp = MCPServer(name="weather", instructions="...")

from weather_mcp import nws
from weather_mcp.config import USER_AGENT
from weather_mcp.errors import WeatherMcpError
from weather_mcp.geocode import get_coordinates


@mcp.tool()
async def get_daily_forecast(address: str, days: int = 3) -> str:
    """Fetches the weather forecast for a plain text city, and or state and or zip code, you can send in an optional days (default 3), and returns a plain text forecast for the next 3 days (day/night)."""
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT}
    ) as client:
        try:
            coordinates = await get_coordinates(client, address)
            forecast = await nws.get_daily_forecast(client, coordinates, days)
            return "\n".join(forecast)
        except WeatherMcpError as e:
            return str(e)

@mcp.tool()
async def get_hourly_forecast(address: str, hours: int = 24) -> str:
    """Fetches the hourly weather forecast for a plain text city, and or state and or zip code, you can send in an optional number of hours (default 24), and returns a plain text forecast for the next n hours."""
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT}
    ) as client:
        try:
            coordinates = await get_coordinates(client, address)
            forecast = await nws.get_hourly_forecast(client, coordinates, hours)
            return "\n".join(forecast)
        except WeatherMcpError as e:
            return str(e)

def main() -> None:
    """Main entry point for the weather-mcp server."""
    mcp.run()
