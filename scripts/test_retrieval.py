import asyncio
import sys

from weather_mcp.rag.db import get_pool
from weather_mcp.rag.embeddings import encode_vectors


async def search(query: str, top_k: int = 5):
    pool = await get_pool()
    query_embedding = encode_vectors([query])[0]
    rows = await pool.fetch(
        """
        SELECT
            issuing_office,
            chunk_type,
            subsection,
            issued_at,
            chunk_text,
            embedding <=> $1 AS distance
        FROM weather_discussion_chunks
        ORDER BY embedding <=> $1
        LIMIT $2
        """,
        query_embedding,
        top_k,
    )
    print(f'Query: "{query}"\n')
    for i, row in enumerate(rows, start=1):
        snippet = " ".join(row["chunk_text"].split())[:160]
        print(f"{i}. [{row['issuing_office']} / {row['chunk_type']} / {row['subsection']}] "
              f"distance={row['distance']:.4f} issued={row['issued_at']}")
        print(f"   {snippet}\n")


if __name__ == "__main__":
    query_text = " ".join(sys.argv[1:]) or "chance of severe thunderstorms and heat advisories"
    asyncio.run(search(query_text))
