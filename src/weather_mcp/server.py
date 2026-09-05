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
from weather_mcp.rag.retrieve import retrieve_chunks

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

@mcp.tool()
async def explain_forecast_reasoning(query: str, office_id: str, top_k: int = 5) -> str:
    """Searches this office's PAST forecast discussions to explain WHY the
    forecast looks the way it does or HOW the forecaster's reasoning changed
    over time (e.g. "why is it so hot right now", "how did the outlook for
    this weekend change this week"). Results are ordered chronologically
    (oldest to newest) so you can trace a change across multiple discussions
    -- this is what makes it different from search_forecast_history, which
    ranks by relevance only and does not preserve time order. Does not
    generate an explanation itself -- cite the returned passages (office,
    date, section) and reason over them directly; don't state anything as
    fact that isn't actually in the returned text.

    office_id is REQUIRED and MUST be a National Weather Service office
    identifier (e.g. "KRAH", "KLIX") -- NOT a city, state, or street address.
    If you don't already know the correct office_id for the location being
    asked about, do not guess one: there is no tool in this server that
    resolves a place name to an office_id, and a wrong guess will silently
    return another office's data. Say you don't have the office code rather
    than fabricating one.

    This searches previously-ingested history, not live data, and only
    covers offices/time periods that happen to have been ingested already --
    it may return nothing even for a real, valid office_id. It is also
    scoped to exactly one office; it cannot trace a pattern (e.g. a heat wave
    or storm system) as it moves across multiple offices' coverage areas --
    use search_forecast_history without an office_id for that instead, and
    expect it to need multiple calls, not one. For today's live forecaster
    discussion rather than search over past ones, use get_weather_discussion
    instead."""
    chunks = await retrieve_chunks(query, office_id=office_id, top_k=top_k, boost_recency=True)
    response = "Relevant forecast reasoning passages:\n\n"
    for chunk in chunks:
        response += f"[{chunk['issuing_office']} / {chunk['chunk_type']} / {chunk['subsection']}] issued={chunk['issued_at']}\n"
        response += f"{chunk['chunk_text']}\n\n"
    return response  # prose-ish, not raw JSON — same as v1

@mcp.tool()
async def search_forecast_history(query: str, office_id: str | None = None, top_k: int = 5) -> str:
    """Semantic search over previously-ingested forecast discussions, ranked
    by topical relevance only (no chronological ordering) -- best for "what
    has been said about X" or tracking a pattern (e.g. a heat wave or storm
    system) as it potentially moves across multiple offices. For "why did
    this change" or "how did the reasoning evolve" questions about ONE
    specific office, use explain_forecast_reasoning instead -- its
    chronological ordering is a better fit and results won't come back
    scattered out of time order. Does not generate an answer itself -- cite
    the returned passages (office, date, section) and reason over them
    directly; don't state anything as fact that isn't actually in the
    returned text.

    office_id is OPTIONAL. If provided, it MUST be a National Weather
    Service office identifier (e.g. "KRAH", "KLIX") -- NOT a city, state, or
    street address. If you don't know the correct office_id for a specific
    location, omit it rather than guessing: a wrong guess silently searches
    the wrong office's data instead of erroring. Omitting office_id searches
    the entire corpus across all ingested offices, which is the right choice
    for cross-country or multi-location questions -- to compare two known
    offices directly, call this tool once per office_id rather than making
    one combined query.

    This searches previously-ingested history, not live data -- it only
    covers offices/time periods that happen to have been ingested already
    and may return nothing even for a real, valid office_id or a real
    weather event. For today's live forecaster discussion rather than search
    over past ones, use get_weather_discussion instead."""
    chunks = await retrieve_chunks(query, office_id=office_id, top_k=top_k, boost_recency=False)
    response = "Relevant forecast passages:\n\n"
    for chunk in chunks:
        response += f"[{chunk['issuing_office']} / {chunk['chunk_type']} / {chunk['subsection']}] issued={chunk['issued_at']}\n"
        response += f"{chunk['chunk_text']}\n\n"
    return response


def main() -> None:
    """Main entry point for the weather-mcp server."""
    mcp.run()
