# Monitoring and Operations

## Runtime Health

The API exposes `/health` for deployment smoke tests and uptime checks.

Expected response:

```json
{
  "status": "ok",
  "models_loaded": ["time_aware_mf", "embedder"],
  "n_recipes": 231637
}
```

`models_loaded` should include:

- `time_aware_mf`: verified production rating model.
- `embedder`: sentence-transformer recipe embeddings used by `/similar`.

## Latency Metrics

The API adds an `X-Process-Time-Ms` response header to every request and exposes
recent in-process latency stats at `/metrics`.

Example:

```json
{
  "model": "time_aware_mf",
  "vector_store": "faiss",
  "latency": {
    "/recommend": {"count": 12, "avg_ms": 241.3, "p95_ms": 308.8, "max_ms": 421.7},
    "/similar": {"count": 12, "avg_ms": 6.4, "p95_ms": 8.2, "max_ms": 9.1}
  }
}
```

The metric store is intentionally lightweight and in-memory. It is useful for smoke
tests, demos, and Cloud Run troubleshooting, but it resets when a container restarts.

## Request Logging

Each request emits a structured log line:

```text
request method=POST path=/recommend status=200 latency_ms=243.17
```

In Cloud Run, these logs are available through Cloud Logging:

```bash
gcloud logging read \
  'resource.type="cloud_run_revision" AND resource.labels.service_name="recipe-recommender"' \
  --project PROJECT_ID \
  --limit 100 \
  --freshness=1h
```

## Cloud Run Cost Controls

The service is configured with `min-instances 0`, so it scales to zero when idle.
The `Cloud Run Control` GitHub Actions workflow can also set `max-instances` to `0`
for a stronger stop switch.

Manual actions:

- `status`: print URL, traffic, memory, and current revision status.
- `stop`: set `min-instances=0` and `max-instances=0`.
- `start`: set `min-instances=0` and `max-instances=RUN_MAX_INSTANCES` or `3`.

## Operational Alerts Worth Adding

Recommended Google Cloud alerts:

- Cloud Run container memory usage above 80%.
- 5xx response count greater than 0 over 5 minutes.
- Request latency p95 above 2 seconds for warm requests.
- Monthly budget threshold alerts at 50%, 75%, and 90%.

## Known Cloud Run Sizing Note

The service loads a 53 MB MF model, a 353 MB recipe embedder, recipe metadata, and
optionally a FAISS index. Runtime memory can exceed 2 GiB during live traffic. The
deployment workflow uses 4 GiB to leave enough headroom.
