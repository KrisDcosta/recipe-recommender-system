# Recipe Recommender: Time-Aware Collaborative Filtering

Rating prediction and top-N recommendation on 1.1M Food.com interactions (231K recipes,
2000–2018). The project prioritises benchmark integrity over model complexity: fixing a
broken evaluation split and documenting zero-rating semantics before any model comparison.

## Results

Zero-rated interactions dropped: Food.com allows reviews without an explicit star rating —
these appear as 0 in the export and are not 1-star reviews. See `FINDINGS.md`.

### Warm-start evaluation (random 70/15/15 split)

| Model | Test RMSE |
|-------|-----------|
| Recipe mean baseline | ~1.24 |
| Linear regression (recipe features) | ~1.18 |
| TF-IDF Ridge (ingredients) | ~1.17 |
| Static MF (SGD, k=10, λ=0.02) | ~0.73 |
| **Time-aware MF (SGD, k=5, λ=0.02)** | **0.69** |

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

## Project Structure

```
.
├── assignment2_1.ipynb     # analysis notebook: EDA → models → evaluation
├── src/
│   ├── data.py             # load_interactions, load_recipes, preprocess, drop_zero_ratings
│   ├── splits.py           # random_split, temporal_split, SplitResult
│   ├── models.py           # BaseMF, StaticMF, TimeAwareMF (fit/predict/save/load)
│   └── metrics.py          # rmse, ndcg_at_k, recall_at_k, bootstrap_ci, paired_ttest
├── tests/                  # pytest suite — 63 tests, no real data required
│   ├── conftest.py         # synthetic fixtures
│   ├── test_data.py
│   ├── test_splits.py
│   ├── test_models.py
│   └── test_metrics.py
├── data/dataset/           # RAW_recipes.csv + RAW_interactions.csv (not in git)
├── pyproject.toml
└── requirements.txt
```

## Setup

```bash
# 1. Install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Download dataset from Kaggle → data/dataset/
# https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions

# 3. Run tests (no dataset needed)
pytest tests/ -v

# 4. Run notebook
jupyter lab assignment2_1.ipynb
```

Run cells top-to-bottom from repo root. Data loading (~30s), Item-CF (~10min), MF training (~15min).

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

## Course

CSE 258R: Recommender Systems & Web Mining · UC San Diego · Fall 2025
