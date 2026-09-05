import asyncio
import json
import logging
from pathlib import Path

import httpx

from weather_mcp.config import USER_AGENT
from weather_mcp.models import Coordinates
from weather_mcp.nws import get_weather_discussion
from weather_mcp.rag.db import get_pool, reset_db
from weather_mcp.rag.ingest import ingest_discussion

logger = logging.getLogger(__name__)


def _load_fixture() -> list[dict]:
    with open(Path(__file__).parent / "test_data.json") as f:
        return json.load(f)


async def seed(strict: bool = True) -> None:
    if strict:
        await reset_db()
    test_data = await asyncio.to_thread(_load_fixture)
    for afd in test_data:
        await ingest_discussion(afd, afd["issuingOffice"], "golden_fixture")
    if not strict:
        await _seed_live_drift()

async def _seed_live_drift() -> None:
    pool = await get_pool()
    offices = await pool.fetch("SELECT office_id, latitude, longitude FROM weather_offices")
    async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}) as client:
        for office in offices:
            coords = Coordinates(lat=office["latitude"], lon=office["longitude"])
            discussions = await get_weather_discussion(client, coords, count = 10)
            for discussion in discussions:
                try:
                    await ingest_discussion(discussion, discussion["issuingOffice"], "live_capture")
                except Exception:
                    logger.exception("drift seed failed for %s", office["office_id"])
if __name__ == "__main__":
    asyncio.run(seed())