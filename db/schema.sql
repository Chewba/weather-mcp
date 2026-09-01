CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- NWS Weather Offices
-- ============================================================

CREATE TABLE weather_offices (
    office_id       VARCHAR(10) PRIMARY KEY,
    office_name     VARCHAR(150) NOT NULL,
    latitude        DOUBLE PRECISION NOT NULL,
    longitude       DOUBLE PRECISION NOT NULL,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- ============================================================
-- Original NWS Product
-- One row per downloaded AFD/product
-- ============================================================

CREATE TABLE weather_discussion_products (
    product_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source              VARCHAR(20) NOT NULL
        CHECK (source IN ('golden_fixture', 'live_capture')),

    issuing_office      VARCHAR(10) NOT NULL
        REFERENCES weather_offices(office_id),

    wmo_collective_id   VARCHAR(20),
    product_code        VARCHAR(20) NOT NULL,
    product_name        VARCHAR(100) NOT NULL,

    issuance_time       TIMESTAMPTZ NOT NULL,

    -- When our application downloaded/stored the product
    retrieved_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    source_url          TEXT NOT NULL,

    -- Keep the original product. Don't rely on chunks
    -- as your only copy of the source.
    raw_product_text    TEXT NOT NULL,

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_product_source_url
        UNIQUE (source_url)
);


-- ============================================================
-- Semantic chunks extracted from the product
-- ============================================================

CREATE TABLE weather_discussion_chunks (
    chunk_id            BIGSERIAL PRIMARY KEY,

    product_id          UUID NOT NULL
        REFERENCES weather_discussion_products(product_id)
        ON DELETE CASCADE,
    source              VARCHAR(20) NOT NULL
        CHECK (source IN ('golden_fixture', 'live_capture')),

    issuing_office      VARCHAR(10) NOT NULL
        REFERENCES weather_offices(office_id),

    -- Denormalized coordinates for retrieval/ranking.
    office_latitude     DOUBLE PRECISION NOT NULL,
    office_longitude    DOUBLE PRECISION NOT NULL,

    -- KEY_MESSAGES, DISCUSSION, AVIATION, MARINE, etc.
    chunk_type          VARCHAR(50) NOT NULL
        CHECK (chunk_type IN ('HEADER', 'KEY_MESSAGES', 'DISCUSSION', 'AVIATION', 'MARINE', 'OTHER')),

    -- Optional description of what this particular chunk covers.
    subsection          VARCHAR(150),

    -- Order in the original product
    chunk_order         INTEGER NOT NULL,

    chunk_text          TEXT NOT NULL,

    -- When the source product was issued
    issued_at           TIMESTAMPTZ NOT NULL,

    -- Approximate period the chunk discusses.
    -- These can be NULL if the period cannot be determined.
    valid_from          TIMESTAMPTZ,
    valid_to            TIMESTAMPTZ,

    -- Optional classification/tagging
    -- e.g. {'heat','wind','thunderstorms'}
    topics              TEXT[],

    embedding           VECTOR(384) NOT NULL,

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_chunk_valid_dates
        CHECK (
            valid_from IS NULL
            OR valid_to IS NULL
            OR valid_to >= valid_from
        ),

    CONSTRAINT uq_product_chunk_order
        UNIQUE (product_id, chunk_order)
);


-- ============================================================
-- Indexes
-- ============================================================

CREATE INDEX idx_weather_discussion_products_office
    ON weather_discussion_products (issuing_office);

CREATE INDEX idx_weather_discussion_products_issuance
    ON weather_discussion_products (issuance_time DESC);


CREATE INDEX idx_weather_discussion_chunks_office
    ON weather_discussion_chunks (issuing_office);

CREATE INDEX idx_weather_discussion_chunks_type
    ON weather_discussion_chunks (chunk_type);

CREATE INDEX idx_weather_discussion_chunks_issued
    ON weather_discussion_chunks (issued_at DESC);

CREATE INDEX idx_weather_discussion_chunks_valid
    ON weather_discussion_chunks (valid_from, valid_to);

CREATE INDEX idx_weather_discussion_chunks_topics
    ON weather_discussion_chunks
    USING GIN (topics);


-- ============================================================
-- Vector similarity index
-- ============================================================

CREATE INDEX idx_weather_discussion_chunks_embedding
    ON weather_discussion_chunks
    USING hnsw (embedding vector_cosine_ops);