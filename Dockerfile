# syntax=docker/dockerfile:1

# --- builder: resolve and install dependencies, then the project itself ---
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install dependencies first, without the project itself, so this layer stays
# cached across source-only changes (only pyproject.toml/uv.lock invalidate it).
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# Now bring in the rest of the source and install the project itself.
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# --- runtime: slim image with just the built venv, no uv/build tooling ---
FROM python:3.12-slim-bookworm

RUN groupadd -r app && useradd -r -g app -d /app app

WORKDIR /app
COPY --from=builder --chown=app:app /app /app

ENV PATH="/app/.venv/bin:$PATH"
USER app

# weather-mcp speaks MCP over stdio, so this container is meant to be run
# with `docker run -i` (interactive stdin/stdout) by an MCP client, not left
# running as a background service -- see README for how to point a client at it.
ENTRYPOINT ["weather-mcp"]
