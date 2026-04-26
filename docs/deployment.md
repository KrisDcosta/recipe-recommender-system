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
7. `/health`, `/predict`, `/recommend`, and `/similar` serve from loaded artifacts.

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
| `MODEL_GCS_URI` | `gs://recipe-recommender-models/models` | Prefix containing `.joblib` artifacts |
| `RECIPES_GCS_URI` | `gs://recipe-recommender-models/data/RAW_recipes.csv` | Recipe metadata CSV |

Required GCS artifacts:

```text
gs://.../models/time_aware_mf.joblib
gs://.../models/user_rated.joblib
gs://.../models/recipe_embedder.joblib    # optional; enables /similar
gs://.../data/RAW_recipes.csv
```

The Cloud Run service account needs read access to the GCS bucket. The deploy service
account needs permissions to push to Artifact Registry and deploy Cloud Run services.

## Alternative: Cloud Build

`cloudbuild.yaml` mirrors the GitHub Actions deploy path for Cloud Build triggers:

```bash
gcloud builds submit \
  --config cloudbuild.yaml \
  --substitutions _REGION=us-central1,_MODEL_GCS_URI=gs://YOUR_BUCKET/models,_RECIPES_GCS_URI=gs://YOUR_BUCKET/data/RAW_recipes.csv
```
