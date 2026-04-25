# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools only in builder stage
RUN pip install --upgrade pip

COPY pyproject.toml .
COPY src/ src/
COPY app/ app/

# Install base + API deps (no dev/embedding extras)
RUN pip install --no-cache-dir -e ".[api]" --prefix=/install


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user for security
RUN useradd --create-home appuser
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ src/
COPY app/ app/

# Models and data are mounted at runtime via docker-compose volumes.
ENV MODEL_DIR=/app/models \
    RECIPES_CSV=/app/data/dataset/RAW_recipes.csv \
    LOG_LEVEL=info \
    PORT=8080

RUN mkdir -p /app/models /app/data/dataset && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --log-level ${LOG_LEVEL}"]
