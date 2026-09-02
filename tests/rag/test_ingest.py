import datetime

import pytest

from weather_mcp import nws
from weather_mcp.rag import ingest

RAW_PRODUCT = {
    "productText": "Area Forecast Discussion text.",
    "productCode": "AFD",
    "issuingOffice": "KRAH",
    "wmoCollectiveId": "FXUS61",
    "@id": "https://api.weather.gov/products/abc123",
    "productName": "Area Forecast Discussion",
    "issuanceTime": "2026-08-31T12:54:00+00:00",
}


def test_parse_afd_feed_extracts_fields():
    result = ingest._parse_afd_feed(RAW_PRODUCT)

    assert result["raw_product_text"] == "Area Forecast Discussion text."
    assert result["product_code"] == "AFD"
    assert result["issuing_office"] == "KRAH"
    assert result["wmo_collective_id"] == "FXUS61"
    assert result["source_url"] == "https://api.weather.gov/products/abc123"
    assert result["product_name"] == "Area Forecast Discussion"
    assert result["issuance_time"] == datetime.datetime(
        2026, 8, 31, 12, 54, tzinfo=datetime.UTC
    )


@pytest.mark.asyncio
async def test_ingest_discussion_runs_pipeline_in_order(monkeypatch):
    calls = []

    async def fake_get_pool():
        return "POOL"

    async def fake_set_office(pool, office_id):
        calls.append(("set_office", pool, office_id))
        return {"office_id": office_id, "latitude": 1.0, "longitude": 2.0}

    async def fake_set_discussion(pool, disc_data):
        calls.append(("set_discussion", pool, disc_data["source_url"]))
        return 99

    def fake_parse_chunks(discussion_data, office_data):
        calls.append(("parse_chunks", discussion_data["product_id"], office_data))
        return ["chunk1", "chunk2"]

    async def fake_upsert_discussion_chunk(pool, chunks):
        calls.append(("upsert_discussion_chunk", pool, chunks))

    monkeypatch.setattr(ingest, "get_pool", fake_get_pool)
    monkeypatch.setattr(ingest, "set_office", fake_set_office)
    monkeypatch.setattr(ingest, "set_discussion", fake_set_discussion)
    monkeypatch.setattr(ingest, "parse_chunks", fake_parse_chunks)
    monkeypatch.setattr(ingest, "upsert_discussion_chunk", fake_upsert_discussion_chunk)

    await ingest.ingest_discussion(RAW_PRODUCT, "KRAH")

    assert calls == [
        ("set_office", "POOL", "KRAH"),
        ("set_discussion", "POOL", "https://api.weather.gov/products/abc123"),
        ("parse_chunks", 99, {"office_id": "KRAH", "latitude": 1.0, "longitude": 2.0}),
        ("upsert_discussion_chunk", "POOL", ["chunk1", "chunk2"]),
    ]


@pytest.mark.asyncio
async def test_get_office_nws_uses_first_station_coordinates(monkeypatch):
    async def fake_get_office_data(client, office_id):
        assert office_id == "RAH"
        return {
            "name": "Raleigh, NC",
            "approvedObservationStations": ["https://api.weather.gov/stations/KRDU"],
        }

    async def fake_get_weather_data(client, url):
        assert url == "https://api.weather.gov/stations/KRDU"
        return {"geometry": {"coordinates": [-78.6382, 35.7796]}}

    monkeypatch.setattr(nws, "get_office_data", fake_get_office_data)
    monkeypatch.setattr(nws, "get_weather_data", fake_get_weather_data)

    result = await ingest.get_office_nws("KRAH")

    assert result == {
        "office_id": "KRAH",
        "longitude": -78.6382,
        "latitude": 35.7796,
        "office_name": "Raleigh, NC",
    }


@pytest.mark.asyncio
async def test_get_office_nws_defaults_when_no_stations(monkeypatch):
    async def fake_get_office_data(client, office_id):
        return {"name": "Raleigh, NC", "approvedObservationStations": []}

    monkeypatch.setattr(nws, "get_office_data", fake_get_office_data)

    result = await ingest.get_office_nws("KRAH")

    assert result == {
        "office_id": "KRAH",
        "longitude": 0,
        "latitude": 0,
        "office_name": "Raleigh, NC",
    }


@pytest.mark.asyncio
async def test_set_office_returns_existing_without_fetching(monkeypatch):
    existing = {"office_id": "KRAH", "office_name": "Raleigh"}

    async def fake_get_office(pool, office_id):
        return existing

    async def fail_get_office_nws(office_id):
        raise AssertionError("should not fetch from NWS when office already exists")

    monkeypatch.setattr(ingest, "get_office", fake_get_office)
    monkeypatch.setattr(ingest, "get_office_nws", fail_get_office_nws)

    result = await ingest.set_office("POOL", "KRAH")

    assert result == existing


@pytest.mark.asyncio
async def test_set_office_creates_when_missing(monkeypatch):
    calls = []
    responses = [None, {"office_id": "KRAH", "office_name": "Raleigh"}]

    async def fake_get_office(pool, office_id):
        return responses.pop(0)

    async def fake_get_office_nws(office_id):
        return {"office_id": office_id, "office_name": "Raleigh", "latitude": 35.7796, "longitude": -78.6382}

    async def fake_upsert_office(pool, office_id, office_name, latitude, longitude):
        calls.append((office_id, office_name, latitude, longitude))

    monkeypatch.setattr(ingest, "get_office", fake_get_office)
    monkeypatch.setattr(ingest, "get_office_nws", fake_get_office_nws)
    monkeypatch.setattr(ingest, "upsert_office", fake_upsert_office)

    result = await ingest.set_office("POOL", "KRAH")

    assert result == {"office_id": "KRAH", "office_name": "Raleigh"}
    assert calls == [("KRAH", "Raleigh", 35.7796, -78.6382)]


@pytest.mark.asyncio
async def test_set_discussion_returns_existing_product_id(monkeypatch):
    async def fake_get_discussion(pool, source_url):
        return {"product_id": 5}

    async def fail_upsert_discussion(*args, **kwargs):
        raise AssertionError("should not insert when discussion already exists")

    monkeypatch.setattr(ingest, "get_discussion", fake_get_discussion)
    monkeypatch.setattr(ingest, "upsert_discussion", fail_upsert_discussion)

    result = await ingest.set_discussion("POOL", {"source_url": "https://example.com/1"})

    assert result == 5


@pytest.mark.asyncio
async def test_set_discussion_inserts_when_missing(monkeypatch):
    responses = [None, {"product_id": 8}]

    async def fake_get_discussion(pool, source_url):
        return responses.pop(0)

    async def fake_upsert_discussion(pool, *args):
        return None

    monkeypatch.setattr(ingest, "get_discussion", fake_get_discussion)
    monkeypatch.setattr(ingest, "upsert_discussion", fake_upsert_discussion)

    disc_data = {
        "source_url": "https://example.com/1",
        "source": "live_capture",
        "issuing_office": "KRAH",
        "wmo_collective_id": "FXUS61",
        "product_code": "AFD",
        "product_name": "Area Forecast Discussion",
        "issuance_time": datetime.datetime(2026, 8, 31, 12, 54, tzinfo=datetime.UTC),
        "raw_product_text": "text",
    }

    result = await ingest.set_discussion("POOL", disc_data)

    assert result == 8
