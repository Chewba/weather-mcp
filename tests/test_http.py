import httpx
import pytest
import respx

import weather_mcp.http as http_mod
from weather_mcp.config import MAX_RETRIES
from weather_mcp.errors import ServiceUnavailableError

NWS_BASE_URL = "https://api.weather.gov"
TEST_URL = f"{NWS_BASE_URL}/points/39.0997,-94.5786"

RESPONSE_BODY = {}


async def fake_sleep(seconds):
    """Replaces asyncio.sleep in retry/backoff tests so they run instantly
    instead of actually waiting out TIME_BASE_RETRY * 2**attempt."""


@respx.mock
@pytest.mark.asyncio
async def test_get_with_retry_success():
    respx.get(TEST_URL).mock(return_value=httpx.Response(200, json=RESPONSE_BODY))
    async with httpx.AsyncClient() as client:
        result = await http_mod.get_with_retry(client, TEST_URL)
    assert result == RESPONSE_BODY


@respx.mock
@pytest.mark.asyncio
async def test_get_with_retry_retries_on_timeout_then_succeeds(monkeypatch):
    monkeypatch.setattr(http_mod.asyncio, "sleep", fake_sleep)
    route = respx.get(TEST_URL)
    route.side_effect = [
        httpx.TimeoutException("timed out"),
        httpx.Response(200, json=RESPONSE_BODY),
    ]

    async with httpx.AsyncClient() as client:
        result = await http_mod.get_with_retry(client, TEST_URL)

    assert result == RESPONSE_BODY
    assert route.call_count == 2


@respx.mock
@pytest.mark.asyncio
async def test_get_with_retry_exhausts_retries_on_persistent_timeout(monkeypatch):
    monkeypatch.setattr(http_mod.asyncio, "sleep", fake_sleep)
    route = respx.get(TEST_URL)
    route.side_effect = httpx.TimeoutException("timed out")

    async with httpx.AsyncClient() as client:
        with pytest.raises(ServiceUnavailableError) as exc_info:
            await http_mod.get_with_retry(client, TEST_URL)

    assert route.call_count == MAX_RETRIES
    assert f"after {MAX_RETRIES} attempts" in str(exc_info.value)


@respx.mock
@pytest.mark.asyncio
async def test_get_with_retry_4xx_raises_immediately_without_retry(monkeypatch):
    monkeypatch.setattr(http_mod.asyncio, "sleep", fake_sleep)
    route = respx.get(TEST_URL).mock(return_value=httpx.Response(404, json={}))

    async with httpx.AsyncClient() as client:
        with pytest.raises(ServiceUnavailableError) as exc_info:
            await http_mod.get_with_retry(client, TEST_URL)

    # a genuine client error (not 429) should fail fast, not burn through retries
    assert route.call_count == 1
    assert "Client error" in str(exc_info.value)


@respx.mock
@pytest.mark.asyncio
async def test_get_with_retry_429_retries_instead_of_raising_immediately(monkeypatch):
    monkeypatch.setattr(http_mod.asyncio, "sleep", fake_sleep)
    route = respx.get(TEST_URL)
    route.side_effect = [
        httpx.Response(429, json={}),
        httpx.Response(200, json=RESPONSE_BODY),
    ]

    async with httpx.AsyncClient() as client:
        result = await http_mod.get_with_retry(client, TEST_URL)

    assert result == RESPONSE_BODY
    assert route.call_count == 2


@respx.mock
@pytest.mark.asyncio
async def test_get_with_retry_5xx_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(http_mod.asyncio, "sleep", fake_sleep)
    route = respx.get(TEST_URL)
    route.side_effect = [
        httpx.Response(503, json={}),
        httpx.Response(200, json=RESPONSE_BODY),
    ]

    async with httpx.AsyncClient() as client:
        result = await http_mod.get_with_retry(client, TEST_URL)

    assert result == RESPONSE_BODY
    assert route.call_count == 2


@respx.mock
@pytest.mark.asyncio
async def test_get_with_retry_waits_out_rate_limit_on_same_host(monkeypatch):
    site = "api.weather.gov"
    monkeypatch.setattr(http_mod.time, "time", lambda: 1000)
    http_mod._next_call[site] = 1005  # pretend the last call means we must wait 5s

    sleep_calls = []

    async def recording_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr(http_mod.asyncio, "sleep", recording_sleep)
    respx.get(TEST_URL).mock(return_value=httpx.Response(200, json=RESPONSE_BODY))

    async with httpx.AsyncClient() as client:
        result = await http_mod.get_with_retry(client, TEST_URL)

    assert result == RESPONSE_BODY
    assert sleep_calls == [5]


@respx.mock
@pytest.mark.asyncio
async def test_get_with_retry_different_hosts_have_independent_rate_limits(monkeypatch):
    # A different host being "due" for a wait should not affect this one --
    # _next_call/_lock are keyed per-site, not shared globally.
    http_mod._next_call["nominatim.openstreetmap.org"] = 99999999999

    sleep_calls = []

    async def recording_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr(http_mod.asyncio, "sleep", recording_sleep)
    respx.get(TEST_URL).mock(return_value=httpx.Response(200, json=RESPONSE_BODY))

    async with httpx.AsyncClient() as client:
        result = await http_mod.get_with_retry(client, TEST_URL)

    assert result == RESPONSE_BODY
    assert sleep_calls == []
