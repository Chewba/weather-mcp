import datetime

import asyncpg
import httpx

from weather_mcp import nws
from weather_mcp.config import USER_AGENT
from weather_mcp.rag.chunking import parse_chunks
from weather_mcp.rag.db import (
    get_discussion,
    get_office,
    get_pool,
    upsert_discussion,
    upsert_discussion_chunk,
    upsert_office,
)


def _parse_afd_feed(raw_text:dict) -> dict:
    response = {}
    response["raw_product_text"] = raw_text["productText"]
    response["product_code"] = raw_text["productCode"]
    response["issuing_office"] = raw_text["issuingOffice"]
    response["wmo_collective_id"] = raw_text["wmoCollectiveId"]
    response["source_url"] = raw_text["@id"]
    response["product_name"] = raw_text["productName"]
    response["issuance_time"] = datetime.datetime.fromisoformat(raw_text["issuanceTime"])
    return response


async def ingest_discussion(raw_text:dict, office_id:str, source:str = "live_capture"):
    pool = await get_pool()
    office_data = await set_office(pool, office_id)
    discussion_data = _parse_afd_feed(raw_text)
    discussion_data['source'] = source
    discussion_data['product_id'] = await set_discussion(pool, discussion_data)
    discussion_data = parse_chunks(discussion_data, office_data)
    await upsert_discussion_chunk(pool, discussion_data)


async def get_office_nws(office_id:str) -> dict:
    async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}) as client:
        office = {
            "office_id" : office_id,
            "longitude" : 0,
            "latitude" : 0,
        }
        offices = await nws.get_office_data(client, office_id[1:])
        office["office_name"] = offices.get("name","")
        station = offices.get("approvedObservationStations",[])
        if len(station) > 0:
            station = await nws.get_weather_data(client, station[0])
            if station is not None:
                station = station.get("geometry",{}).get("coordinates",[0,0])
                office["longitude"] = station[0]
                office["latitude"] = station[1]
        return office

async def set_office(pool:asyncpg.Pool, office_id:str):
    office_data = await get_office(pool, office_id)
    if office_data is None:
        office_data = await get_office_nws(office_id)
        await upsert_office(pool, office_id, office_data["office_name"], office_data["latitude"], office_data["longitude"])
        office_data = await get_office(pool, office_id)
    return office_data

async def set_discussion(pool:asyncpg.Pool, disc_data:dict) -> int:
    discussion = await get_discussion(pool, disc_data["source_url"])
    time_now = datetime.datetime.now(datetime.UTC)
    if discussion is not None:
        return discussion["product_id"]
    discussion = await upsert_discussion(pool, disc_data["source"], disc_data["issuing_office"], disc_data["wmo_collective_id"], disc_data["product_code"], disc_data["product_name"], disc_data["issuance_time"], time_now, disc_data["source_url"], disc_data["raw_product_text"])
    discussion = await get_discussion(pool, disc_data["source_url"])
    return discussion["product_id"]