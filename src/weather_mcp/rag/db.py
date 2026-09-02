import asyncpg
from pgvector.asyncpg import register_vector

from weather_mcp import config
from weather_mcp.rag.embeddings import encode_vectors


async def _init_conn(conn):
    await register_vector(conn)

_pool: asyncpg.Pool | None = None

async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            database=config.DB_NAME,
            host=config.DB_HOST,
            port=config.DB_PORT,
            command_timeout=60,
            init=_init_conn,
        )
    return _pool

async def reset_db() -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            TRUNCATE TABLE weather_discussion_chunks
                        , weather_discussion_products;
            """
        )

async def upsert_office(
    pool: asyncpg.Pool,
    office_id: str,
    office_name: str,
    latitude: float,
    longitude: float,
) -> None:
    await pool.execute(
        """
        INSERT INTO weather_offices (office_id, office_name, latitude, longitude)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (office_id) DO NOTHING
        """,
        office_id,
        office_name,
        latitude,
        longitude,
    )

async def get_office(pool:asyncpg.Pool, office_id: str) -> asyncpg.Record:
    office_data = await pool.fetchrow(
        """
        select office_id, office_name, latitude, longitude, created_at, updated_at
            from weather_offices where office_id = $1
        """,
        office_id,
    )
    return office_data

async def upsert_discussion(
    pool: asyncpg.Pool,
    source:str,
    issuing_office:str,
    wmo_collective_id:str,
    product_code:str,
    product_name:str,
    issuance_time:str,
    retrieved_at:str,
    source_url:str,
    raw_product_text:str,
) -> None:
    response = await pool.fetchval(
        """
        INSERT INTO weather_discussion_products (source, issuing_office, wmo_collective_id, product_code, product_name, issuance_time, retrieved_at, source_url, raw_product_text)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        ON CONFLICT (source_url) DO NOTHING
        RETURNING product_id
        """,
        source,
        issuing_office,
        wmo_collective_id,
        product_code,
        product_name,
        issuance_time,
        retrieved_at,
        source_url,
        raw_product_text,
    )
    return response

async def get_discussion(pool:asyncpg.Pool, source_url: str) -> asyncpg.Record:
    discussion_data = await pool.fetchrow(
        """
            SELECT product_id, source, issuing_office, wmo_collective_id, product_code, product_name, issuance_time, retrieved_at, source_url, raw_product_text, created_at
            FROM weather_discussion_products
            WHERE source_url = $1
        """,
        source_url,
    )
    return discussion_data

async def upsert_discussion_chunk(
    pool: asyncpg.Pool,
    chunks: list[dict]
) -> None:
    chunk_tuple = []
    for chunk in chunks:
        chunk_tuple.append((
            chunk["product_id"],
            chunk["source"],
            chunk["issuing_office"],
            chunk["office_latitude"],
            chunk["office_longitude"],
            chunk["chunk_type"],
            chunk["subsection"],
            chunk["chunk_order"],
            chunk["chunk_text"],
            chunk["issued_at"],
            chunk["valid_from"],
            chunk["valid_to"],
            chunk["topics"],
            chunk["embedding"],
        ))
    response = await pool.executemany(
        """
        INSERT INTO weather_discussion_chunks (    
            product_id,
            source,
            issuing_office,
            office_latitude,
            office_longitude,
            chunk_type,
            subsection,
            chunk_order,
            chunk_text,
            issued_at,
            valid_from,
            valid_to,
            topics,
            embedding
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
        ON CONFLICT (product_id, chunk_order) DO NOTHING
        """,
        chunk_tuple,
    )
    return response

async def get_discussion_chunk(pool:asyncpg.Pool, chunk_id: int) -> asyncpg.Record:
    discussion_data = await pool.fetchrow(
        """
            SELECT
                chunk_id, 
                product_id,
                source,
                issuing_office,
                office_latitude,
                office_longitude,
                chunk_type,
                subsection,
                chunk_order,
                chunk_text,
                issued_at,
                valid_from,
                valid_to,
                topics,
                embedding,
                created_at
            FROM weather_discussion_chunks
            WHERE chunk_id = $1
        """,
        chunk_id,
    )
    return discussion_data

async def search_chunks(query: str, office_id: str | None = None, top_k: int = 5) -> list[asyncpg.Record]:
    pool = await get_pool()
    query_embedding = encode_vectors([query])[0]
    where_clause = "WHERE issuing_office = $3" if office_id else ""
    sql = f"""
            SELECT
                issuing_office,
                chunk_type,
                subsection,
                issued_at,
                chunk_text,
                embedding <=> $1 AS distance
            FROM weather_discussion_chunks
            {where_clause}
            ORDER BY embedding <=> $1
            LIMIT $2
        """
    if office_id:
        rows = await pool.fetch(sql, query_embedding, top_k, office_id)
    else:
        rows = await pool.fetch(sql, query_embedding, top_k)
    return rows
