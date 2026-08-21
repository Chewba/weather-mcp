import pytest

from weather_mcp.errors import LocationAmbiguousError, LocationNotFoundError
from weather_mcp.geocode import get_coordinates


@pytest.mark.asyncio
async def test_get_coordinates_parses_raw_lat_lon_without_network(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("a raw lat/lon pair should never hit the network")

    monkeypatch.setattr("weather_mcp.geocode.get_with_retry", fail_if_called)

    coordinates = await get_coordinates(client=None, address="36.07264,-79.79198")

    assert coordinates.lat == 36.0726
    assert coordinates.lon == -79.792


@pytest.mark.asyncio
async def test_get_coordinates_zip_code_single_result(monkeypatch):
    async def fake_get_with_retry(client, url, **kwargs):
        return [{"lat": "36.0956", "lon": "-79.8269"}]

    monkeypatch.setattr("weather_mcp.geocode.get_with_retry", fake_get_with_retry)

    coordinates = await get_coordinates(client=None, address="27401")

    assert coordinates.lat == 36.0956
    assert coordinates.lon == -79.8269


@pytest.mark.asyncio
async def test_get_coordinates_free_text_single_result(monkeypatch):
    async def fake_get_with_retry(client, url, **kwargs):
        return [{"lat": "36.0726", "lon": "-79.7919"}]

    monkeypatch.setattr("weather_mcp.geocode.get_with_retry", fake_get_with_retry)

    coordinates = await get_coordinates(client=None, address="Greensboro, NC")

    assert coordinates.lat == 36.0726
    assert coordinates.lon == -79.7919


@pytest.mark.asyncio
async def test_get_coordinates_no_results_raises_location_not_found(monkeypatch):
    async def fake_get_with_retry(client, url, **kwargs):
        return []

    monkeypatch.setattr("weather_mcp.geocode.get_with_retry", fake_get_with_retry)

    with pytest.raises(LocationNotFoundError):
        await get_coordinates(client=None, address="Nowhereville")


@pytest.mark.asyncio
async def test_get_coordinates_ambiguous_resolved_by_full_state_name(monkeypatch):
    async def fake_get_with_retry(client, url, **kwargs):
        return [
            {"lat": "39.7817", "lon": "-89.6501", "address": {"state": "Illinois", "ISO3166-2-lvl4": "US-IL"}},
            {"lat": "42.1015", "lon": "-72.5898", "address": {"state": "Massachusetts", "ISO3166-2-lvl4": "US-MA"}},
        ]

    monkeypatch.setattr("weather_mcp.geocode.get_with_retry", fake_get_with_retry)

    coordinates = await get_coordinates(client=None, address="Springfield, Illinois")

    assert coordinates.lat == 39.7817
    assert coordinates.lon == -89.6501


@pytest.mark.asyncio
async def test_get_coordinates_ambiguous_resolved_by_state_abbreviation(monkeypatch):
    async def fake_get_with_retry(client, url, **kwargs):
        return [
            {"lat": "39.7817", "lon": "-89.6501", "address": {"state": "Illinois", "ISO3166-2-lvl4": "US-IL"}},
            {"lat": "42.1015", "lon": "-72.5898", "address": {"state": "Massachusetts", "ISO3166-2-lvl4": "US-MA"}},
        ]

    monkeypatch.setattr("weather_mcp.geocode.get_with_retry", fake_get_with_retry)

    coordinates = await get_coordinates(client=None, address="Springfield, IL")

    assert coordinates.lat == 39.7817
    assert coordinates.lon == -89.6501


@pytest.mark.asyncio
async def test_get_coordinates_ambiguous_without_match_raises_with_states(monkeypatch):
    async def fake_get_with_retry(client, url, **kwargs):
        return [
            {"lat": "39.7817", "lon": "-89.6501", "address": {"state": "Illinois", "ISO3166-2-lvl4": "US-IL"}},
            {"lat": "42.1015", "lon": "-72.5898", "address": {"state": "Massachusetts", "ISO3166-2-lvl4": "US-MA"}},
        ]

    monkeypatch.setattr("weather_mcp.geocode.get_with_retry", fake_get_with_retry)

    with pytest.raises(LocationAmbiguousError) as exc_info:
        await get_coordinates(client=None, address="Springfield")

    assert exc_info.value.states == ["Illinois", "Massachusetts"]
