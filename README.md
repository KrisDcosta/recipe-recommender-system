# Recipe Recommender: Time-Aware Collaborative Filtering + LLM Embeddings

[![CI](https://github.com/KrisDcosta/CSE258_A2/actions/workflows/ci.yml/badge.svg)](https://github.com/KrisDcosta/CSE258_A2/actions/workflows/ci.yml)
[![Deploy to Cloud Run](https://github.com/KrisDcosta/CSE258_A2/actions/workflows/deploy.yml/badge.svg)](https://github.com/KrisDcosta/CSE258_A2/actions/workflows/deploy.yml)

Rating prediction and top-N recommendation on 1.1M Food.com interactions (231K recipes,
2000–2018). The project has two threads: (1) benchmark integrity — fixing a broken
evaluation split and verifying zero-rating semantics; (2) model progression from global
mean → time-aware MF → hybrid MF + LLM content embeddings.

## Results

Zero-rated interactions dropped: Food.com allows reviews without an explicit star rating —
these appear as 0 in the export and are not 1-star reviews. See [`FINDINGS.md`](FINDINGS.md)
for the full benchmark-integrity narrative.

### Verified warm-start evaluation (random 70/15/15 split)

These values are from the current reproducible pipeline after dropping Food.com zero-rating
rows and splitting with seed 42.

| Model | Test RMSE | NDCG@10 | Recall@10 |
|-------|----------:|--------:|----------:|
| Global mean baseline | 0.7251 | - | - |
| Recipe mean baseline | 0.7677 | - | - |
| Time-aware MF (SGD, k=5, λ=0.02) | **0.6834** | **0.2252** | **0.3153** |

The verified CLI run is exported in [`results/phase3_metrics.json`](results/phase3_metrics.json).
The current production service loads this MF artifact plus the recipe embedder used by
`/similar`; a committed `hybrid_mf.joblib` rating artifact is not part of the current
deployment.

### Data and split

| Quantity | Value |
|----------|------:|
| Raw interactions | 1,132,367 |
| Raw recipes | 231,637 |
| Zero-rated rows dropped | 60,847 |
| Zero-rated share | 5.4% |
| Train rows | 750,064 |
| Validation rows | 160,728 |
| Test rows | 160,728 |
| Best validation RMSE | 0.6808 |

### Cold-start breakdown for deployed model

| Item bucket | Test rows | RMSE |
|-------------|----------:|-----:|
| Cold, fewer than 5 training interactions | 67,249 | 0.7195 |
| Medium, 5-19 training interactions | 45,498 | 0.6426 |
| Warm, at least 20 training interactions | 47,981 | 0.6688 |

### Statistical validation (Section 7)

The notebook includes the bootstrap/paired-test framework for model comparisons. The
repo-visible production metrics above are the concrete values verified by the current
CLI/API/Docker/Cloud Run path.

## Key Findings

- **Broken benchmark diagnosed**: Kaggle split had no recipe overlap — item-CF collapsed to global mean predictions. Random split restored overlap and validated CF models.
- **Zero ratings verified**: 60,847 entries (5.4%) are 0. Food.com confirmed: no star submitted ≠ 1-star review. Both pipelines documented.
- **Rating drift is real**: average rating fell ~0.4 points 2002→2014, partially recovered — time bins capture signal static MF misses.
- **Time-aware improvement is statistically significant**: bootstrap CI and paired t-test confirm the gap is not sampling noise.
- **LLM embeddings add semantic content signal**: all-MiniLM-L6-v2 on recipe name + ingredients captures similarity that bag-of-words can miss.
- **Hybrid architecture targets cold-start**: sparse recipes can use content embeddings when collaborative history is weak or unavailable.
- **Production path verified**: CLI training writes model artifacts, FastAPI serves predictions, and Docker Compose returns `/health` with the trained model loaded.

## Architecture

![Recipe recommender architecture](docs/architecture.png)

Source: [`docs/architecture.excalidraw`](docs/architecture.excalidraw)

## Project Structure

```
.
├── assignment2_1.ipynb     # analysis notebook: EDA → models → evaluation
├── FINDINGS.md             # benchmark integrity, zero-rating, drift, cold-start narrative
├── .github/workflows/      # CI and Cloud Run deployment workflows
├── app/                    # FastAPI inference service
│   ├── main.py             # startup model loading, /health
│   ├── schemas.py          # Pydantic request/response schemas
│   └── routers/            # /predict, /recommend, /similar
├── scripts/                # reproducible training/evaluation CLIs
│   ├── train.py
│   ├── evaluate.py
│   └── embed_recipes.py
├── src/
│   ├── data.py             # load_interactions, load_recipes, preprocess, drop_zero_ratings
│   ├── splits.py           # random_split, temporal_split, SplitResult
│   ├── models.py           # BaseMF, StaticMF, TimeAwareMF (fit/predict/save/load)
│   ├── metrics.py          # rmse, ndcg_at_k, recall_at_k, bootstrap_ci, paired_ttest
│   ├── embeddings.py       # RecipeEmbedder (sentence-transformers), build_embedding_features
│   └── hybrid.py           # HybridMF = TimeAwareMF + LLM embeddings → Ridge
├── tests/                  # pytest suite — no real data required
│   ├── conftest.py         # synthetic fixtures
│   ├── test_data.py
│   ├── test_splits.py
│   ├── test_models.py
│   ├── test_metrics.py
│   ├── test_embeddings.py
│   └── test_hybrid.py
├── results/
│   └── phase3_metrics.json # committed summary of verified CLI/API/Docker run
├── docs/
│   ├── architecture.excalidraw
│   ├── architecture.png
│   └── deployment.md       # CI/CD, Cloud Run, and GCS artifact setup
├── models/                 # saved model artifacts (not in git)
├── data/dataset/           # RAW_recipes.csv + RAW_interactions.csv (not in git)
├── Dockerfile
├── docker-compose.yml
├── cloudbuild.yaml
├── pyproject.toml
└── requirements.txt
```

## Setup

```bash
# 1. Install base dependencies
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. (Optional) Install LLM embedding dependencies
pip install -e ".[embeddings]"

# 3. Download dataset from Kaggle → data/dataset/
# https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions

# 4. Run tests (no dataset needed)
pytest tests/ -v

# 5. Run notebook
jupyter lab assignment2_1.ipynb
```

Run cells top-to-bottom from repo root. Data loading (~30s), Item-CF (~10min), MF training (~15min),
LLM embedding (~2–3 min on CPU, cached after first run).

## Production Pipeline

Train and evaluate from the command line:

```bash
python scripts/train.py --model hybrid --data-dir data/dataset --output-dir models
python scripts/evaluate.py --model models/hybrid_mf.joblib --data-dir data/dataset --output results/hybrid_eval.json
python scripts/embed_recipes.py --data-dir data/dataset --output models/recipe_embedder.joblib
```

The training pipeline writes the MF model artifacts plus `models/user_rated.joblib`,
which the API uses to exclude recipes already rated by a user. The recipe embedder is
generated separately and enables `/similar`.

Run the API locally:

```bash
uvicorn app.main:app --reload --port 8000
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"user_id": 123, "recipe_id": 456, "date": "2015-06"}'
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 123, "top_n": 10, "exclude_rated": true}'
curl -X POST http://localhost:8000/similar \
  -H "Content-Type: application/json" \
  -d '{"recipe_id": 456, "top_n": 5}'
```

Run with Docker Compose after training artifacts exist in `models/`:

```bash
docker compose up --build
curl http://localhost:8080/health
```

Verified Phase 3 serving stack:

```text
pytest: 143 passed
FastAPI /health: 200 OK
Docker Compose /health: 200 OK
Loaded model artifacts: time_aware_mf, embedder
```

## CI/CD and Cloud Run

Live API:

```text
https://recipe-recommender-tyhw3omfqq-uc.a.run.app
```

Smoke test:

```bash
curl https://recipe-recommender-tyhw3omfqq-uc.a.run.app/health
curl -X POST https://recipe-recommender-tyhw3omfqq-uc.a.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{"user_id": 123, "recipe_id": 456, "date": "2015-06"}'
curl -X POST https://recipe-recommender-tyhw3omfqq-uc.a.run.app/similar \
  -H "Content-Type: application/json" \
  -d '{"recipe_id": 456, "top_n": 2}'
```

Verified `/similar` response:

```json
{
  "seed_recipe_id": 456,
  "similar": [
    {"recipe_id": 153501, "name": "easy dal", "similarity": 0.9415},
    {"recipe_id": 81727, "name": "yellow lentil dal", "similarity": 0.9411}
  ]
}
```

GitHub Actions workflows live in `.github/workflows/`:

- `ci.yml`: runs package imports, pytest, and Docker image build checks on pull requests and pushes to `main`.
- `deploy.yml`: builds the runtime image, pushes it to Artifact Registry, and deploys to Cloud Run on pushes to `main`.

Cloud Run loads both MF and embedder artifacts from GCS at startup when local mounted
files are absent. `/health` reports `time_aware_mf` and `embedder` when both artifacts
are available.
See [`docs/deployment.md`](docs/deployment.md) for required GitHub secrets, repository
variables, GCS artifact layout, and the optional Cloud Build path.

## Notebook Sections

| Section | Content |
|---------|---------|
| 1. Data Loading & Preprocessing | Load, parse nutrition, merge, 70/15/15 split |
| EDA | Rating distributions, 18-year time trend, cold-start analysis |
| 2. Baseline Models | Recipe mean, Item-CF, linear regression, TF-IDF Ridge, static MF |
| 3. Time-Aware MF | SGD with user×time and item×time bias terms |
| 4. Results Comparison | RMSE table, all models |
| 5. Temporal Evaluation | Train < 2015 / test ≥ 2015, cold-start breakdown, RMSE gap |
| 6. Ranking Metrics | NDCG@10, Recall@10, sampled negatives, static vs time-aware |
| 7. A/B Statistical Framework | Bootstrap CI, paired t-test, Cohen's d, would-deploy decision |
| 8. LLM Semantic Embeddings | all-MiniLM-L6-v2 embeddings, LLM Ridge, Hybrid MF+LLM, similar-recipe explorer |

## Course

CSE 258R: Recommender Systems & Web Mining · UC San Diego · Fall 2025
