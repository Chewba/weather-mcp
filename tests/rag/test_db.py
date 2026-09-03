import pytest

from weather_mcp.rag import db


class FakePool:
    def __init__(self, fetch_result=None, fetchrow_result=None, fetchval_result=None):
        self.fetch_result = fetch_result if fetch_result is not None else []
        self.fetchrow_result = fetchrow_result
        self.fetchval_result = fetchval_result
        self.calls = []

    async def fetch(self, sql, *args):
        self.calls.append(("fetch", sql, args))
        return self.fetch_result

    async def fetchrow(self, sql, *args):
        self.calls.append(("fetchrow", sql, args))
        return self.fetchrow_result

    async def fetchval(self, sql, *args):
        self.calls.append(("fetchval", sql, args))
        return self.fetchval_result

    async def execute(self, sql, *args):
        self.calls.append(("execute", sql, args))

    async def executemany(self, sql, args_list):
        self.calls.append(("executemany", sql, args_list))


@pytest.fixture
def fake_pool(monkeypatch):
    pool = FakePool()

    async def fake_get_pool():
        return pool

    monkeypatch.setattr(db, "get_pool", fake_get_pool)
    return pool


@pytest.fixture(autouse=True)
def fake_encode_vectors(monkeypatch):
    monkeypatch.setattr(db, "encode_vectors", lambda texts: [[0.1, 0.2, 0.3]])


@pytest.mark.asyncio
async def test_upsert_office_passes_correct_args(fake_pool):
    await db.upsert_office(fake_pool, "KRAH", "Raleigh", 35.7796, -78.6382)

    kind, sql, args = fake_pool.calls[0]
    assert kind == "execute"
    assert "INSERT INTO weather_offices" in sql
    assert args == ("KRAH", "Raleigh", 35.7796, -78.6382)


@pytest.mark.asyncio
async def test_get_office_returns_row(fake_pool):
    fake_pool.fetchrow_result = {"office_id": "KRAH"}

    result = await db.get_office(fake_pool, "KRAH")

    assert result == {"office_id": "KRAH"}
    kind, _sql, args = fake_pool.calls[0]
    assert kind == "fetchrow"
    assert args == ("KRAH",)


@pytest.mark.asyncio
async def test_upsert_discussion_returns_product_id(fake_pool):
    fake_pool.fetchval_result = 7

    result = await db.upsert_discussion(
        fake_pool, "live_capture", "KRAH", "TISRAH", "AFD", "Area Forecast Discussion",
        "2026-08-31T12:54:00+00:00", "2026-08-31T13:00:00+00:00",
        "https://api.weather.gov/products/abc123", "raw text",
    )

    assert result == 7
    kind, sql, args = fake_pool.calls[0]
    assert kind == "fetchval"
    assert "ON CONFLICT (source_url) DO NOTHING" in sql
    assert args[-2:] == ("https://api.weather.gov/products/abc123", "raw text")


@pytest.mark.asyncio
async def test_upsert_discussion_chunk_preserves_column_order(fake_pool):
    """Regression guard: the INSERT column list and the tuple built from each
    chunk dict must stay in the same order, or values silently land in the
    wrong columns."""
    chunk = {
        "product_id": 1,
        "source": "live_capture",
        "issuing_office": "KRAH",
        "office_latitude": 35.7796,
        "office_longitude": -78.6382,
        "chunk_type": "HEADER",
        "subsection": None,
        "chunk_order": 0,
        "chunk_text": "text",
        "issued_at": "2026-08-31T12:54:00+00:00",
        "valid_from": None,
        "valid_to": None,
        "topics": None,
        "embedding": [0.1, 0.2, 0.3],
    }

    await db.upsert_discussion_chunk(fake_pool, [chunk])

    kind, _sql, args_list = fake_pool.calls[0]
    assert kind == "executemany"
    assert args_list == [(
        1, "live_capture", "KRAH", 35.7796, -78.6382, "HEADER", None, 0,
        "text", "2026-08-31T12:54:00+00:00", None, None, None, [0.1, 0.2, 0.3],
    )]


@pytest.mark.asyncio
async def test_search_chunks_with_office_id_filters_and_parameterizes(fake_pool):
    fake_pool.fetch_result = [{"issuing_office": "KRAH"}]

    result = await db.search_chunks("hot and humid", office_id="KRAH", top_k=5)

    assert result == [{"issuing_office": "KRAH"}]
    kind, sql, args = fake_pool.calls[0]
    assert kind == "fetch"
    assert "WHERE issuing_office = $3" in sql
    assert args == ([0.1, 0.2, 0.3], 5, "KRAH")


@pytest.mark.asyncio
async def test_search_chunks_without_office_id_has_no_leftover_placeholder(fake_pool):
    """Regression guard for the earlier `.replace()`-templated SQL, which left
    a literal "/issue_office/" token (or a dangling WHERE clause) in the query
    whenever office_id was None, breaking every unscoped call."""
    await db.search_chunks("hot and humid", office_id=None, top_k=5)

    kind, sql, args = fake_pool.calls[0]
    assert kind == "fetch"
    assert "issue_office" not in sql
    assert "WHERE" not in sql
    assert args == ([0.1, 0.2, 0.3], 5)


@pytest.mark.asyncio
async def test_filter_new_chunks_drops_exact_text_duplicate(fake_pool):
    fake_pool.fetch_result = [
        {"chunk_type": "AVIATION", "subsection": "AVIATION", "chunk_text": "VFR conditions expected."}
    ]

    new_chunk = {
        "issuing_office": "KBOU",
        "chunk_type": "AVIATION",
        "subsection": "AVIATION",
        "chunk_text": "VFR conditions expected.",
    }

    result = await db.filter_new_chunks(fake_pool, [new_chunk])

    assert result == []
    kind, sql, args = fake_pool.calls[0]
    assert kind == "fetch"
    assert "WHERE issuing_office = $1" in sql
    assert args == ("KBOU",)


@pytest.mark.asyncio
async def test_filter_new_chunks_ignores_whitespace_and_case_differences(fake_pool):
    """NWS reissues often carry the same text with trivial whitespace
    reflow -- normalization must catch these as duplicates too, not just
    byte-identical text."""
    fake_pool.fetch_result = [
        {"chunk_type": "AVIATION", "subsection": "AVIATION", "chunk_text": "VFR   conditions\nexpected."}
    ]

    new_chunk = {
        "issuing_office": "KBOU",
        "chunk_type": "AVIATION",
        "subsection": "AVIATION",
        "chunk_text": "vfr conditions expected.",
    }

    result = await db.filter_new_chunks(fake_pool, [new_chunk])

    assert result == []


@pytest.mark.asyncio
async def test_filter_new_chunks_keeps_genuinely_new_text(fake_pool):
    fake_pool.fetch_result = [
        {"chunk_type": "AVIATION", "subsection": "AVIATION", "chunk_text": "VFR conditions expected."}
    ]

    new_chunk = {
        "issuing_office": "KBOU",
        "chunk_type": "AVIATION",
        "subsection": "AVIATION",
        "chunk_text": "IFR conditions with low ceilings expected.",
    }

    result = await db.filter_new_chunks(fake_pool, [new_chunk])

    assert result == [new_chunk]


@pytest.mark.asyncio
async def test_filter_new_chunks_keeps_same_text_in_a_different_subsection(fake_pool):
    """A duplicate check scoped only to (office, chunk_type) -- ignoring
    subsection -- would wrongly drop two genuinely different key messages
    that happen to share wording. subsection must be part of the key."""
    fake_pool.fetch_result = [
        {"chunk_type": "KEY_MESSAGES", "subsection": "1", "chunk_text": "Heat continues."}
    ]

    new_chunk = {
        "issuing_office": "KBOU",
        "chunk_type": "KEY_MESSAGES",
        "subsection": "2",
        "chunk_text": "Heat continues.",
    }

    result = await db.filter_new_chunks(fake_pool, [new_chunk])

    assert result == [new_chunk]


@pytest.mark.asyncio
async def test_filter_new_chunks_empty_input_short_circuits(fake_pool):
    result = await db.filter_new_chunks(fake_pool, [])

    assert result == []
    assert fake_pool.calls == []


@pytest.mark.asyncio
async def test_search_chunks_embeds_the_query_text(fake_pool, monkeypatch):
    captured = {}

    def fake_encode(texts):
        captured["texts"] = texts
        return [[9.0, 9.0, 9.0]]

    monkeypatch.setattr(db, "encode_vectors", fake_encode)

    await db.search_chunks("severe thunderstorms", top_k=3)

    assert captured["texts"] == ["severe thunderstorms"]
    _, _, args = fake_pool.calls[0]
    assert args[0] == [9.0, 9.0, 9.0]
