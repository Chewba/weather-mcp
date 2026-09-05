import datetime

import pytest

from weather_mcp.rag import retrieve


def _row(office, distance, issued_at, chunk_type="DISCUSSION", subsection=None):
    return {
        "issuing_office": office,
        "chunk_type": chunk_type,
        "subsection": subsection,
        "issued_at": issued_at,
        "chunk_text": f"text for {office} at {issued_at}",
        "distance": distance,
    }


def _dt(day, hour):
    return datetime.datetime(2026, 8, day, hour, tzinfo=datetime.UTC)


@pytest.mark.asyncio
async def test_retrieve_chunks_without_boost_uses_top_k_directly(monkeypatch):
    captured = {}

    async def fake_search_chunks(query, office_id=None, top_k=5):
        captured["args"] = (query, office_id, top_k)
        return [_row("KRAH", 0.52, _dt(31, 10)), _row("KRAH", 0.55, _dt(30, 5))]

    monkeypatch.setattr(retrieve, "search_chunks", fake_search_chunks)

    results = await retrieve.retrieve_chunks("hot and humid", office_id="KRAH", top_k=5, boost_recency=False)

    assert captured["args"] == ("hot and humid", "KRAH", 5)
    # no re-sort/truncation: order and count match whatever search_chunks returned
    assert [r["distance"] for r in results] == [0.52, 0.55]


@pytest.mark.asyncio
async def test_retrieve_chunks_boost_recency_widens_candidate_pool(monkeypatch):
    captured = {}

    async def fake_search_chunks(query, office_id=None, top_k=5):
        captured["top_k"] = top_k
        return []

    monkeypatch.setattr(retrieve, "search_chunks", fake_search_chunks)

    await retrieve.retrieve_chunks("hot and humid", office_id="KRAH", top_k=5, boost_recency=True)

    assert captured["top_k"] == 5 * retrieve.RECENCY_CANDIDATE_MULTIPLIER


@pytest.mark.asyncio
async def test_retrieve_chunks_boost_recency_picks_most_recent_and_orders_oldest_first(monkeypatch):
    """Regression test: boost_recency must pick the most-recent chunks from
    the (wider) relevance-filtered candidate pool, then present them
    oldest-first so a calling model can read them as a causal timeline -
    not just re-sort whatever the raw top-k happened to contain."""
    candidates = [
        _row("KRAH", 0.60, _dt(29, 8)),
        _row("KRAH", 0.53, _dt(31, 23)),
        _row("KRAH", 0.58, _dt(30, 6)),
        _row("KRAH", 0.51, _dt(31, 18)),
        _row("KRAH", 0.62, _dt(29, 14)),
        _row("KRAH", 0.55, _dt(31, 10)),
    ]

    async def fake_search_chunks(query, office_id=None, top_k=5):
        return candidates

    monkeypatch.setattr(retrieve, "search_chunks", fake_search_chunks)

    results = await retrieve.retrieve_chunks("hot and humid", office_id="KRAH", top_k=3, boost_recency=True)

    assert len(results) == 3
    assert [r["issued_at"] for r in results] == [_dt(31, 10), _dt(31, 18), _dt(31, 23)]


@pytest.mark.asyncio
async def test_retrieve_chunks_returns_dicts_with_expected_fields(monkeypatch):
    async def fake_search_chunks(query, office_id=None, top_k=5):
        return [_row("KRAH", 0.52, _dt(31, 10), chunk_type="KEY_MESSAGES", subsection="1")]

    monkeypatch.setattr(retrieve, "search_chunks", fake_search_chunks)

    results = await retrieve.retrieve_chunks("hot and humid", top_k=5, boost_recency=False)

    assert results == [{
        "issuing_office": "KRAH",
        "chunk_type": "KEY_MESSAGES",
        "subsection": "1",
        "issued_at": _dt(31, 10),
        "chunk_text": f"text for KRAH at {_dt(31, 10)}",
        "distance": 0.52,
    }]
