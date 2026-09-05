from weather_mcp.rag.db import search_chunks

# When boost_recency is on, pull a wider relevance-ranked candidate pool before
# re-sorting by time, so recency ordering picks among genuinely relevant chunks
# instead of collapsing to whichever product happens to be newest.
RECENCY_CANDIDATE_MULTIPLIER = 4

# v3 adaptive-retrieval fix for the documented "one office's vocabulary
# dominates the raw top-k" failure (see FINDINGS_RAG.md's multi-hop causality
# case): an unscoped, cross-office query pulls a much wider candidate pool,
# then caps how many chunks any single office can contribute, so an office
# that happens to phrase the query's topic less similarly still gets a shot
# at appearing instead of being crowded out entirely.
DIVERSIFY_CANDIDATE_MULTIPLIER = 20
DIVERSIFY_PER_OFFICE_CAP = 3

async def retrieve_chunks(
    query: str,
    office_id: str | None = None,
    top_k: int = 5,
    boost_recency: bool = True,
    diversify_offices: bool = False,
) -> list[dict]:
    diversify_offices = diversify_offices and office_id is None
    if diversify_offices:
        candidate_k = top_k * DIVERSIFY_CANDIDATE_MULTIPLIER
    elif boost_recency:
        candidate_k = top_k * RECENCY_CANDIDATE_MULTIPLIER
    else:
        candidate_k = top_k
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
    if diversify_offices:
        results = _diversify_by_office(results, top_k, DIVERSIFY_PER_OFFICE_CAP)
    elif boost_recency:
        results.sort(key=lambda c: c["issued_at"])
        results = results[-top_k:]
    return results


def _diversify_by_office(results: list[dict], top_k: int, per_office_cap: int) -> list[dict]:
    """results is already distance-sorted, best first. Walk it in that order,
    skipping a chunk once its office has already contributed per_office_cap
    chunks, so one dominant-vocabulary office can't fill every slot."""
    counts: dict[str, int] = {}
    diversified = []
    for chunk in results:
        office = chunk["issuing_office"]
        if counts.get(office, 0) >= per_office_cap:
            continue
        counts[office] = counts.get(office, 0) + 1
        diversified.append(chunk)
        if len(diversified) >= top_k:
            break
    return diversified
