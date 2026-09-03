"""Recall@k benchmark for search_forecast_history's underlying retrieval
(rag/retrieve.py -> rag/db.py's embedding <=> query cosine search).

Unlike the case studies in tests/eval/FINDINGS_RAG.md (eyeball the top-5 for
one query), this computes an actual number: of all the chunks that are truly
relevant to a query, what fraction show up in the top-k results, for
k = 5/10/20/50? A ground-truth *relevant set* is required for that -- each
BENCHMARK_CASE below defines one via a SQL WHERE clause, using the same
"scan chunk_text directly, don't trust retrieval to tell you the ground
truth" discipline already used throughout FINDINGS_RAG.md (e.g. the "161
ground-truth ridge/heat chunks" figure for the multi-hop causality case).

Ground truth is computed live against whatever's in the DB right now, not
hardcoded -- chunk_ids are BIGSERIAL and not stable across reseeds, and
`--corpus-mode drift` runs mean row counts can shift between runs anyway.
Run `docker ps` first; this needs the DB up, same as any other eval script.

BENCHMARK_CASES below were authored against the original 2-3-day-dense
fixture and its curated ridge-tracking narrative (a subtropical ridge
moving KFWD -> ... -> KRAH). scripts/build_test_data.py later rebuilt the
fixture across real distinct days, and that narrative's records were
merged back in (deduped by product id) rather than left behind -- see
tests/eval/FINDINGS_RAG.md's "Corpus overhaul" section. Re-verified live
after the merge: all 7 of the case's offices still carry the original
ridge/heat content at the original dates. Ground truth counts will still
drift a little run to run as the daily-sampled portion of the fixture is
regenerated, since that always reflects whatever's live in NWS at the time.

Usage:
    uv run python scripts/recall_at_k.py
"""

import asyncio

from weather_mcp.rag.db import get_pool
from weather_mcp.rag.embeddings import encode_vectors

TOP_KS = (5, 10, 20, 50)

BENCHMARK_CASES = [
    {
        "name": "ridge/heat multi-hop chain (KFWD -> KRAH)",
        "query": "a ridge of high pressure aloft building and shifting "
        "eastward bringing hot temperatures",
        # Ground truth per FINDINGS_RAG.md's multi-hop causality case: every
        # chunk from an office genuinely on the documented ridge path,
        # mentioning the ridge or the heat it brought. Restricting to these
        # 7 offices (not a blanket ILIKE over the whole corpus) avoids
        # counting an unrelated heat/ridge mention from an unrelated office
        # as a false ground-truth positive.
        "ground_truth_where": """
            issuing_office IN ('KFWD', 'KSHV', 'KMEG', 'KOHX', 'KFFC', 'KCAE', 'KRAH')
            AND (chunk_text ILIKE '%ridge%' OR chunk_text ILIKE '%heat%')
        """,
    },
    {
        "name": "office-scoped topical search (KLIX severe wx / heat advisories)",
        "query": "chance of severe thunderstorms and heat advisories",
        "ground_truth_where": """
            issuing_office = 'KLIX'
            AND (
                chunk_text ILIKE '%severe%'
                OR chunk_text ILIKE '%heat advisor%'
                OR chunk_text ILIKE '%thunderstorm%'
            )
        """,
    },
]


async def ground_truth_chunk_ids(pool, where_clause: str) -> set[int]:
    rows = await pool.fetch(
        f"SELECT chunk_id FROM weather_discussion_chunks WHERE {where_clause}"
    )
    return {row["chunk_id"] for row in rows}


async def retrieved_chunk_ids(pool, query: str, top_k: int) -> list[int]:
    query_embedding = encode_vectors([query])[0]
    rows = await pool.fetch(
        """
        SELECT chunk_id
        FROM weather_discussion_chunks
        ORDER BY embedding <=> $1
        LIMIT $2
        """,
        query_embedding,
        top_k,
    )
    return [row["chunk_id"] for row in rows]


async def run_case(pool, case: dict) -> None:
    truth_ids = await ground_truth_chunk_ids(pool, case["ground_truth_where"])
    print(f"\n{case['name']}")
    print(f'  query: "{case["query"]}"')
    print(f"  ground-truth relevant chunks: {len(truth_ids)}")

    if not truth_ids:
        print("  (empty ground-truth set -- recall@k is undefined here, skipping. "
              "This shape of case, e.g. a negative control, needs a different "
              "metric: whether retrieval's top-1 distance stays high/low-confidence, "
              "not recall.)")
        return

    for k in TOP_KS:
        result_ids = await retrieved_chunk_ids(pool, case["query"], k)
        hits = len(set(result_ids) & truth_ids)
        recall = hits / len(truth_ids)
        print(f"  recall@{k:<3} = {recall:6.1%}  ({hits}/{len(truth_ids)} relevant chunks in top-{k})")


async def main() -> None:
    pool = await get_pool()
    for case in BENCHMARK_CASES:
        await run_case(pool, case)


if __name__ == "__main__":
    asyncio.run(main())
