import re

import httpx

from weather_mcp.errors import LocationAmbiguousError, LocationNotFoundError
from weather_mcp.http import get_with_retry
from weather_mcp.models import Coordinates

_LAT_LON_REGEX = re.compile(r"^[-+]?([1-8]?\d(\.\d+)?|90(\.0+)?),\s*[-+]?(180(\.0+)?|((1[0-7]\d)|([1-9]?\d))(\.\d+)?)$")
_ZIP_REGEX = re.compile(r"^\d{5}(-\d{4})?$")

async def get_coordinates(client:httpx.AsyncClient, address: str) -> Coordinates:
    """Fetches the geographical coordinates for the given address using the Nominatim API."""
    if match := _LAT_LON_REGEX.match(address):
        return Coordinates(lat=round(float(match.group(1)), 4), lon=round(float(match.group(4)), 4))
    elif match := _ZIP_REGEX.match(address):
        params = {"postalcode": address, "limit": 1, "country": "us"}
        return await _nominatim_call(client, params, address)
    params = {"q": address, "limit": 5}
    return await _nominatim_call(client, params, address)

async def _nominatim_call(client:httpx.AsyncClient, params: dict, origin_address:str) -> Coordinates:
    """Fetches the geographical coordinates for the given address using the Nominatim API."""
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        **params,
        "format": "jsonv2",
        "addressdetails": 1
    }
    response = await get_with_retry(client, url, params=params)
    data = response
    if not data:
        raise LocationNotFoundError(f"No coordinates found for address: {origin_address}")
    elif len(data) == 1:
        lat = round(float(data[0]["lat"]), 4)
        lon = round(float(data[0]["lon"]), 4)
        return Coordinates(lat=lat, lon=lon)
    org_state = origin_address.split(",")[-1].strip().upper()
    states = []
    for item in data:
        state = item.get("address", {}).get("state", "")
        state_abbr = item.get("address", {}).get("ISO3166-2-lvl4", "").upper()
        if org_state == state.upper() or org_state == f"US-{state_abbr}":
            lat = round(float(item["lat"]), 4)
            lon = round(float(item["lon"]), 4)
            return Coordinates(lat=lat, lon=lon)
        states.append(state)
    raise LocationAmbiguousError(states=states)
