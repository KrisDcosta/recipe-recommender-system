# Operational Monitoring Evidence

Captured: April 28, 2026.

## Cloud Monitoring Dashboard

- Dashboard: Recipe Recommender - Cloud Run Operations
- Dashboard resource: `projects/544327697237/dashboards/c27d1740-9183-4bdf-aa2a-caff3c0604bf`
- Console link: https://console.cloud.google.com/monitoring/dashboards/custom/c27d1740-9183-4bdf-aa2a-caff3c0604bf?project=noted-gift-494503
- Reproducible config: [`cloud-run-dashboard.json`](cloud-run-dashboard.json)

The dashboard tracks:

- Request rate by response class.
- Request latency p95.
- Container memory utilization.
- Instance count.
- Operational runbook notes.

## Live Service Status

```text
URL: https://recipe-recommender-tyhw3omfqq-uc.a.run.app
Ready: True
Revision: recipe-recommender-00011-t58
Traffic: 100% latest revision
CPU: 1
Memory: 4Gi
Container concurrency: 80
Min instances: 0
Max instances: 1
LLM explanations: disabled; rule-based fallback active
```

## Live Health Check

```json
{
  "status": "ok",
  "models_loaded": ["time_aware_mf", "embedder"],
  "n_recipes": 231637
}
```

## Live Latency Snapshot

The API exposes in-process route latency at `/metrics`. This resets on container restart,
so it is runtime evidence rather than a durable analytics store.

```json
{
  "model": "time_aware_mf",
  "vector_store": "faiss",
  "latency": {
    "/health": {"count": 1, "avg_ms": 4.39, "p95_ms": 4.39, "max_ms": 4.39},
    "/explain": {"count": 1, "avg_ms": 5.68, "p95_ms": 5.68, "max_ms": 5.68}
  }
}
```

## Recent Cloud Logging Evidence

Recent request logs showed successful live traffic for:

- `GET /health` with `200 OK`.
- `GET /metrics` with `200 OK`.
- `POST /recommend/new-user` with `200 OK`.
- `GET /demo` and browser-triggered `GET /health` checks.

Useful log queries:

```bash
gcloud logging read \
  'resource.type="cloud_run_revision" AND resource.labels.service_name="recipe-recommender"' \
  --project noted-gift-494503 \
  --limit 50 \
  --freshness=24h

gcloud logging read \
  'resource.type="cloud_run_revision" AND resource.labels.service_name="recipe-recommender" AND httpRequest.status>=500' \
  --project noted-gift-494503 \
  --limit 50 \
  --freshness=24h
```

## Operational Criteria

The service is operational when:

- `/health` returns `status=ok` and includes both `time_aware_mf` and `embedder`.
- `/metrics` reports `vector_store=faiss`.
- Cloud Run service status is Ready.
- Cloud Run traffic is 100% to the latest revision.
- Request logs show 2xx responses for `/recommend`, `/recommend/new-user`, `/similar`, `/explain`, `/metrics`, and `/demo`.

## Cost Control

The service uses `min-instances=0` and can be manually stopped with the `Cloud Run Control`
GitHub Actions workflow. The stop action sets `max-instances=0`; the start action restores
`max-instances=3`.
