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

Current live evidence is captured in
[`docs/operations/monitoring-evidence.md`](operations/monitoring-evidence.md), with a
visual screenshot at [`docs/operations/monitoring-evidence.png`](operations/monitoring-evidence.png).

## Google Cloud Monitoring Dashboard

A Cloud Monitoring dashboard has been created for the deployed service:

```text
Recipe Recommender - Cloud Run Operations
projects/544327697237/dashboards/c27d1740-9183-4bdf-aa2a-caff3c0604bf
```

Console link:

```text
https://console.cloud.google.com/monitoring/dashboards/custom/c27d1740-9183-4bdf-aa2a-caff3c0604bf?project=noted-gift-494503
```

The dashboard definition is versioned in
[`docs/operations/cloud-run-dashboard.json`](operations/cloud-run-dashboard.json). It
contains widgets for request rate by response class, p95 latency, memory utilization,
instance count, and a short runbook note.

To recreate the dashboard:

```bash
gcloud monitoring dashboards create \
  --project noted-gift-494503 \
  --config-from-file docs/operations/cloud-run-dashboard.json
```

To list matching dashboards:

```bash
gcloud monitoring dashboards list \
  --project noted-gift-494503 \
  --format='value(name,displayName)' | rg 'Recipe Recommender'
```

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

5xx-only query:

```bash
gcloud logging read \
  'resource.type="cloud_run_revision" AND resource.labels.service_name="recipe-recommender" AND httpRequest.status>=500' \
  --project PROJECT_ID \
  --limit 50 \
  --freshness=24h
```

## Cloud Run Cost Controls

The service is configured with `min-instances 0`, so it scales to zero when idle.
The `Cloud Run Control` GitHub Actions workflow can also set `max-instances` to `0`
for a stronger stop switch.

Manual actions:

- `status`: print URL, traffic, memory, and current revision status.
- `stop`: set `min-instances=0` and `max-instances=0`.
- `start`: set `min-instances=0` and `max-instances=RUN_MAX_INSTANCES` or `1`.

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
