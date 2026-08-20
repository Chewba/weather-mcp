class WeatherMcpError(Exception):
    """Base class for all WeatherMCP errors."""

class ServiceUnavailableError(WeatherMcpError):
    """Raised when the NWS API service is unavailable."""
    
class LocationNotFoundError(WeatherMcpError):
    """Raised when the specified location is not found."""

class LocationAmbiguousError(WeatherMcpError):
    """Raised when the specified location is ambiguous and cannot be resolved to a single set of coordinates."""
    def __init__(self, message: str = "The specified location is ambiguous and cannot be resolved to a single set of coordinates.", states: list[str] | None = None):
        states = states or []
        self.message = message
        self.states = states
        super().__init__(f"{self.message} States: {', '.join(self.states)}")