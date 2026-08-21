import pytest

from weather_mcp import http


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    http._lock.clear()
    http._next_call.clear()
