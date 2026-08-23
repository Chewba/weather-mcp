from importlib.metadata import metadata

_meta = metadata("weather-mcp")
USER_AGENT = (
    f"{_meta['Name']}/{_meta['Version']} ({_meta['Project-URL'].split(', ', 1)[1]})"
)
MAX_RETRIES = 3
