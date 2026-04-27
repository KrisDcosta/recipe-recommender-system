# Deployment

## Phase 4A: CI

`.github/workflows/ci.yml` runs on pull requests and pushes to `main`.

Jobs:

- `Tests and imports`: installs `.[dev,api]`, imports app/scripts/package modules, and runs `pytest`.
- `Docker build`: builds the runtime Docker image without requiring model artifacts.

This keeps CI independent of the Food.com dataset and trained `.joblib` files.

## Phase 4B: Cloud Run

`.github/workflows/deploy.yml` runs on pushes to `main` and manual dispatch.

Deployment flow:

1. Authenticate GitHub Actions to Google Cloud with Workload Identity Federation.
2. Build the `runtime` Docker image.
3. Push the image to Artifact Registry.
4. Deploy the image to Cloud Run.
5. Cloud Run starts the FastAPI app.
6. At startup, the app downloads model/data artifacts from GCS if local files are absent.
7. `/health`, `/predict`, `/recommend`, `/similar`, `/metrics`, and `/demo` serve from loaded artifacts.
8. The workflow smoke-tests `/health`, `/recommend`, `/similar`, and `/metrics` after deployment and fails if Cloud Run is not reachable.

Required GitHub Actions secrets:

| Name | Purpose |
|------|---------|
| `GCP_PROJECT_ID` | Google Cloud project ID |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | Workload Identity Provider resource name |
| `GCP_SERVICE_ACCOUNT` | Deploying service account email |

Required GitHub Actions variables:

| Name | Example | Purpose |
|------|---------|---------|
| `GCP_REGION` | `us-central1` | Cloud Run and Artifact Registry region |
| `GAR_REPOSITORY` | `recipe-recommender` | Artifact Registry Docker repository |
| `RUN_SERVICE_ACCOUNT` | `recipe-recommender-run@PROJECT_ID.iam.gserviceaccount.com` | Runtime identity used by Cloud Run to read GCS artifacts |
| `MODEL_GCS_URI` | `gs://recipe-recommender-models/models` | Prefix containing `.joblib` artifacts |
| `MODEL_NAME` | `time_aware_mf` | Rating model loaded by the API; defaults to the verified production model |
| `RECIPES_GCS_URI` | `gs://recipe-recommender-models/data/RAW_recipes.csv` | Recipe metadata CSV |

Required GCS artifacts:

```text
gs://.../models/time_aware_mf.joblib
gs://.../models/user_rated.joblib
gs://.../models/recipe_embedder.joblib    # enables /similar
gs://.../data/RAW_recipes.csv
```

The API defaults to `MODEL_NAME=time_aware_mf` because that is the verified best
full-data model. `hybrid_mf.joblib` can be uploaded and loaded with
`MODEL_NAME=hybrid_mf`, but the April 27, 2026 full-data run underperformed
time-aware MF and should not be the production default.

Cloud Run needs enough memory for the MF model, the 353 MB recipe embedder, and the
recipe lookup table at the same time. The deploy workflow uses `--memory 4Gi`; the
previous 2 GiB service exceeded its memory limit under live requests on April 27, 2026.
When deployment is healthy, `/health` reports both `time_aware_mf` and `embedder`.

The Cloud Run service account needs read access to the GCS bucket. The deploy service
account needs permissions to push to Artifact Registry and deploy Cloud Run services.

## Alternative: Cloud Build

`cloudbuild.yaml` mirrors the GitHub Actions deploy path for Cloud Build triggers:

```bash
gcloud builds submit \
  --config cloudbuild.yaml \
  --substitutions _REGION=us-central1,_MODEL_GCS_URI=gs://YOUR_BUCKET/models,_RECIPES_GCS_URI=gs://YOUR_BUCKET/data/RAW_recipes.csv
```

## Start, Stop, and Status

Use the `Cloud Run Control` GitHub Actions workflow for manual operations:

- `status`: prints service URL, traffic split, revision status, memory, and autoscaling annotations.
- `stop`: sets `min-instances=0` and `max-instances=0`.
- `start`: sets `min-instances=0` and `max-instances` back to `RUN_MAX_INSTANCES` or `3`.

The equivalent local commands are:

```bash
gcloud run services update recipe-recommender \
  --project PROJECT_ID \
  --region us-central1 \
  --min-instances 0 \
  --max-instances 0

gcloud run services update recipe-recommender \
  --project PROJECT_ID \
  --region us-central1 \
  --min-instances 0 \
  --max-instances 3
```
