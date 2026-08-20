import httpx
import asyncio
import time
from weather_mcp.config import MAX_RETRIES
from weather_mcp.errors import ServiceUnavailableError
TIME_BASE_RETRY = 0.5
TIME_TO_NEXT_CALL = 1
_lock: dict[str, asyncio.Lock] = {}
_next_call: dict[str, int] = {}

async def get_with_retry(client:httpx.AsyncClient, url:str, **kwargs) -> dict:
    """Fetches data from the given URL with retry logic."""
    site = url.split("/")[2]
    async with _lock.setdefault(site, asyncio.Lock()):
        time_now = int(time.time())
        if site in _next_call and time_now < _next_call[site]:
            await asyncio.sleep(_next_call[site] - time_now)
            time_now = int(time.time())
        _next_call[site] = time_now + TIME_TO_NEXT_CALL
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.get(url, **kwargs)
                response.raise_for_status()
                return response.json()
            except (httpx.TimeoutException, httpx.TransportError) as e:
                await asyncio.sleep(TIME_BASE_RETRY * 2 ** attempt)
            except httpx.HTTPStatusError as e:
                if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                    raise ServiceUnavailableError(f"Client error occurred while trying to reach {url}: {e}")
                await asyncio.sleep(TIME_BASE_RETRY * 2 ** attempt)
        raise ServiceUnavailableError(f"Failed to fetch data from {url} after {MAX_RETRIES} attempts.")