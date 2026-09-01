import os
from importlib.metadata import metadata

from dotenv import load_dotenv

load_dotenv()

_meta = metadata("weather-mcp")
USER_AGENT = (
    f"{_meta['Name']}/{_meta['Version']} ({_meta['Project-URL'].split(', ', 1)[1]})"
)
MAX_RETRIES = 3

DB_USER = os.environ.get("WEATHER_MCP_DB_USER", "weather-mcp-user")
DB_PASSWORD = os.environ.get("WEATHER_MCP_DB_PASSWORD", "testing-weather-mcp-user")
DB_NAME = os.environ.get("WEATHER_MCP_DB_NAME", "weather-mcp")
DB_HOST = os.environ.get("WEATHER_MCP_DB_HOST", "localhost")
DB_PORT = int(os.environ.get("WEATHER_MCP_DB_PORT", "5432"))
