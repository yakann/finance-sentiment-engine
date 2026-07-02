# syntax=docker/dockerfile:1
# Stage 1: Install dependencies with uv
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS builder

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Stage 2: Slim runtime image
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY . .

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
