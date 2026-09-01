# weather-mcp

An MCP server that turns the National Weather Service (NWS) API into simple, LLM-friendly tools for forecasts, current conditions, alerts, and forecaster discussions.

## What it does

Six tools, all accepting a plain-text US city/state or ZIP code:

- **`get_current_conditions`** — the weather right now, from the nearest reporting station.
- **`get_daily_forecast`** — a multi-day outlook (day/night periods), up to whatever NWS actually provides (see Known limitations).
- **`get_hourly_forecast`** — an hour-by-hour forecast for near-term, specific-time questions.
- **`get_active_alerts`** — active NWS alerts (advisories, warnings) for a location.
- **`get_weather_discussion`** — the human forecaster's own technical narrative behind the forecast, for "why" questions.
- **`compare_forecasts`** — compares the daily forecast for two locations in one call.

## Design principles

- **Pydantic models** (`models.py`) validate and normalize coordinates before they ever reach an API call.
- **A small error hierarchy** (`errors.py`) — `LocationNotFoundError`, `LocationAmbiguousError`, `ServiceUnavailableError` — lets the tool layer in `server.py` catch failures at the boundary and return a graceful plain-text message instead of raising through to the MCP client.
- **Retry/backoff with per-host rate limiting** (`http.py`) wraps every outbound call, since both Nominatim (geocoding) and api.weather.gov have their own usage limits.
- **src-layout package** (`src/weather_mcp/`), tested with `pytest` against the geocoding and NWS logic independently of any live network call (fixtures + `monkeypatch`, no recorded-HTTP dependency).

## Setup

```
uv sync
```

## Running the server

```
uv run weather-mcp
```

Registered as the `weather-mcp` console script (see `pyproject.toml`), speaking MCP over stdio.

## Running with Docker

A multi-stage `Dockerfile` is included (`uv`-based builder stage, slim non-root runtime stage). **Build Verified** — docker build completes cleanly and the entrypoint starts and exits 0 with no traceback.

```
docker build -t weather-mcp .
```

Since this server speaks MCP over stdio, it's meant to be run interactively by an MCP client, not left running in the background — `docker run -d` would just see an immediately-closed stdin and exit. To point an MCP client (e.g. Claude Desktop's config) at the container instead of a local `uv run weather-mcp`, use something like:

```json
{
  "mcpServers": {
    "weather": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "weather-mcp"]
    }
  }
}
```

## RAG extension (in progress)

A retrieval layer is being added on top of the existing tools, backed by Postgres + `pgvector`, so questions like "why did the forecast change" can be answered by retrieving relevant historical forecaster discussions, not just the current one.

Current state:
- **`docker-compose.yml`** — runs `pgvector/pgvector:pg16` locally, with a bind-mounted `pgdata/` volume (gitignored) for persistence.
- **`db/schema.sql`** — auto-runs on first container start via `docker-entrypoint-initdb.d`. Defines `weather_offices`, `weather_discussion_products` (raw AFD text, one row per downloaded product), and `weather_discussion_chunks` (semantic chunks with 384-dim embeddings and an HNSW cosine-similarity index). Both product and chunk rows carry a `source` (`golden_fixture` vs `live_capture`) so evals can run against a frozen fixture set or against real captured data.

To bring the DB up locally:

```
docker compose up -d
docker compose ps                 # wait for "healthy"
docker exec -it weather-mcp-db-1 psql -U weather-mcp-user -d weather-mcp -c "\dt"
```

Not yet built: a seed script to load a checked-in golden fixture into the DB, the two new retrieval-facing MCP tools (`search_forecast_history`, `explain_forecast_reasoning`), and the corresponding `eval-strict`/`eval-with-drift` extension to the eval harness below.

## Testing

```
uv run pytest
uv run ruff check .
```

Both also run automatically in CI (`.github/workflows/ci.yml`) on every push and pull request.

## Eval harness

`tests/eval/` is a from-scratch evaluation harness for the model-facing side of this server — not just "does the code work," but "does an LLM actually pick the right tool, with the right parameters, and give an honest, well-grounded answer." It's the part of this project that mattered most to build carefully; the full methodology and findings are written up in [`tests/eval/FINDINGS.md`](tests/eval/FINDINGS.md).

Briefly:
- **`questions.py`** — 16 natural-language test questions, each with one or more acceptable tool-call *strategies* (not just one "correct" answer), scored with weights, optional parameters, and mutual exclusivity for redundant alternatives (e.g. calling `compare_forecasts` vs. calling `get_daily_forecast` twice).
- **`grading.py`** — fully deterministic, no LLM involved — scores which tools got called correctly, independent of answer quality.
- **`run_eval.py`** — runs the suite against two backends: headless (`claude -p`, billed against a Claude subscription) or direct API (`anthropic.messages.create()` with a real multi-turn tool loop, billed against `ANTHROPIC_API_KEY`) — plus an LLM judge (averaged over multiple samples) that scores the final answer's quality independently of tool selection, and per-question/total cost tracking for both the model under test and the judge.

Highlights from `FINDINGS.md`: the eval process caught several real server bugs no unit test had (a broken point-data lookup, a corrupted no-alerts message, mismatched grid-ID resolution), demonstrated that a single LLM judge sample isn't reliable on its own (the same prompt scored a 2 and a 9 on separate runs), and caught a model confidently fabricating an entire week's forecast that didn't exist in the data it was given.

## Known limitations

- **US locations only** — bounded by NWS's own coverage; a non-USA address gets no useful answer.
- **Forecast horizon caps at 7 days** regardless of what's requested — this is an NWS API limit, not something this server can extend.
