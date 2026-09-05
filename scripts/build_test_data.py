"""Regenerates scripts/test_data.json from live NWS data, spread across
real calendar days instead of the previous 2-3-day dense-reissuance window.

Why: the previous fixture (archived as test_data_ridge_case_archive.json)
took 10 discussions per office within a ~2-3 day span, which meant most of
those 10 were near-identical reissuances of the same underlying AFD -- see
tests/eval/FINDINGS_RAG.md's "Duplicate-chunk corpus finding" (47.1% of the
whole corpus was exact-text duplicate chunks before rag/db.py's
filter_new_chunks was added). Spreading samples across real distinct days
gives genuinely different content per sample instead of copies.

Constraint, verified live before writing this (see conversation): NWS's
`https://api.weather.gov/products/types/AFD/locations/{loc}` only exposes a
rolling ~6-7 day window and accepts no start/end/limit query params -- so
this pulls 1 discussion per available day (closest to 18:00 UTC, a
mid-afternoon-everywhere-in-CONUS proxy since office-local "midday" would
need per-office timezone handling this doesn't do), not a full 10 days.

Usage:
    uv run python scripts/build_test_data.py
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

import httpx

from weather_mcp.config import USER_AGENT
from weather_mcp.nws import get_weather_data

# Same 16 offices as the previous fixture (scripts/test_data_ridge_case_archive.json).
OFFICES = [
    "KBOU", "KBOX", "KCAE", "KFFC", "KFWD", "KLIX", "KLOT", "KLOX",
    "KMEG", "KMFL", "KMPX", "KOHX", "KPSR", "KRAH", "KSEW", "KSHV",
]

MID_DAY_UTC_HOUR = 18


def _pick_one_per_day(products: list[dict]) -> list[dict]:
    by_date: dict[str, list[dict]] = {}
    for p in products:
        issued = datetime.fromisoformat(p["issuanceTime"])
        by_date.setdefault(issued.date().isoformat(), []).append(p)

    picked = []
    for _date, day_products in sorted(by_date.items()):
        def distance_from_midday(p: dict) -> float:
            issued = datetime.fromisoformat(p["issuanceTime"])
            midday = issued.replace(hour=MID_DAY_UTC_HOUR, minute=0, second=0, microsecond=0)
            return abs((issued - midday).total_seconds())

        picked.append(min(day_products, key=distance_from_midday))
    return picked


async def build_office(client: httpx.AsyncClient, office_id: str) -> list[dict]:
    location = office_id[1:]
    listing = await get_weather_data(
        client, f"https://api.weather.gov/products/types/AFD/locations/{location}"
    )
    products = listing.get("@graph", [])
    selected = _pick_one_per_day(products)
    print(f"{office_id}: {len(products)} available, "
          f"{len(selected)} distinct days selected "
          f"({selected[0]['issuanceTime'] if selected else 'n/a'} .. "
          f"{selected[-1]['issuanceTime'] if selected else 'n/a'})")

    full_records = []
    for entry in selected:
        record = await get_weather_data(client, entry["@id"])
        full_records.append(record)
    return full_records


async def main() -> None:
    async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, timeout=30) as client:
        all_records = []
        for office_id in OFFICES:
            all_records.extend(await build_office(client, office_id))

    out_path = Path(__file__).parent / "test_data.json"

    def _write():
        with open(out_path, "w") as f:
            json.dump(all_records, f, indent=2)

    await asyncio.to_thread(_write)
    print(f"\nWrote {len(all_records)} records to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
