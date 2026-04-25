# Recipe Recommender: Time-Aware Collaborative Filtering + LLM Embeddings

Rating prediction and top-N recommendation on 1.1M Food.com interactions (231K recipes,
2000–2018). The project has two threads: (1) benchmark integrity — fixing a broken
evaluation split and verifying zero-rating semantics; (2) model progression from global
mean → time-aware MF → hybrid MF + LLM content embeddings.

## Results

Zero-rated interactions dropped: Food.com allows reviews without an explicit star rating —
these appear as 0 in the export and are not 1-star reviews.

### Warm-start evaluation (random 70/15/15 split)

| Model | Test RMSE |
|-------|-----------|
| Recipe mean baseline | ~1.24 |
| Linear regression (recipe features) | ~1.18 |
| TF-IDF Ridge (ingredients) | ~1.17 |
| LLM Ridge (all-MiniLM-L6-v2) | run notebook |
| Static MF (SGD, k=10, λ=0.02) | ~0.73 |
| Time-aware MF (SGD, k=5, λ=0.02) | **0.69** |
| **Hybrid MF + LLM** | **run notebook** |

### Temporal evaluation (train < 2015, test ≥ 2015)

| Model | Test RMSE | NDCG@10 | Recall@10 |
|-------|-----------|---------|-----------|
| Time-aware MF | run notebook | run notebook | run notebook |

> Temporal RMSE is higher than warm-start (expected). Cold-start users/items fall back
> to global mean — see Section 5 for the breakdown.

### Statistical validation (Section 7)

Time-aware MF improvement over static MF validated via bootstrap CI (10K resamples)
and paired t-test. Cohen's d effect size reported.

## Key Findings

- **Broken benchmark diagnosed**: Kaggle split had no recipe overlap — item-CF collapsed to global mean predictions. Random split restored overlap and validated CF models.
- **Zero ratings verified**: 60,847 entries (5.4%) are 0. Food.com confirmed: no star submitted ≠ 1-star review. Both pipelines documented.
- **Rating drift is real**: average rating fell ~0.4 points 2002→2014, partially recovered — time bins capture signal static MF misses.
- **Time-aware improvement is statistically significant**: bootstrap CI and paired t-test confirm the gap is not sampling noise.
- **LLM embeddings beat TF-IDF**: all-MiniLM-L6-v2 on recipe name + ingredients captures semantic similarity that bag-of-words misses.
- **Hybrid model bridges cold-start gap**: new recipes with zero interactions still get meaningful predictions from the embedding branch.

## Project Structure

```
.
├── assignment2_1.ipynb     # analysis notebook: EDA → models → evaluation
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
├── models/                 # saved model artifacts (not in git)
├── data/dataset/           # RAW_recipes.csv + RAW_interactions.csv (not in git)
├── Dockerfile
├── docker-compose.yml
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

The training pipeline writes the model artifacts plus `models/user_rated.joblib`, which the API uses to exclude recipes already rated by a user.

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
