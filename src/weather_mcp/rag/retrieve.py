from weather_mcp.rag.db import search_chunks

# When boost_recency is on, pull a wider relevance-ranked candidate pool before
# re-sorting by time, so recency ordering picks among genuinely relevant chunks
# instead of collapsing to whichever product happens to be newest.
RECENCY_CANDIDATE_MULTIPLIER = 4

async def retrieve_chunks(query: str, office_id: str | None = None, top_k: int = 5, boost_recency: bool = True) -> list[dict]:
    candidate_k = top_k * RECENCY_CANDIDATE_MULTIPLIER if boost_recency else top_k
    rows = await search_chunks(query, office_id=office_id, top_k=candidate_k)
    results = [
        {
            "issuing_office": row["issuing_office"],
            "chunk_type": row["chunk_type"],
            "subsection": row["subsection"],
            "issued_at": row["issued_at"],
            "chunk_text": row["chunk_text"],
            "distance": row["distance"],
        }
        for row in rows
    ]
    if boost_recency:
        results.sort(key=lambda c: c["issued_at"])
        results = results[-top_k:]
    return results
