# Findings: Benchmark Integrity and Recommender Evaluation

This project started as a recommendation benchmark, but the main result was diagnostic:
the naive benchmark shape did not actually test collaborative filtering. The final
pipeline treats benchmark construction as part of the model, not a preprocessing detail.

## 1. Broken Benchmark: No Recipe Overlap

The original Kaggle-provided split was structurally hostile to item-based
collaborative filtering because train and test recipes did not overlap. That means
the model had no item history for test recipes, so item-CF could not use collaborative
signal and collapsed toward global mean predictions.

The fix was not to make the model more complex. The fix was to make the evaluation
match the question:

- Use a random 70/15/15 interaction split for warm-start rating prediction.
- Track user and item overlap explicitly through `SplitResult.overlap_stats()`.
- Keep temporal splitting as a separate deployment-style stress test.

This distinction matters because warm-start and temporal/cold-start evaluation answer
different questions. A single RMSE number without split diagnostics is not enough.

## 2. Zero Ratings Are Missing Ratings, Not One-Star Ratings

The raw interaction export contains 60,847 zero-rated rows, about 5.4% of the
1,132,367 raw interactions. These are not treated as one-star preferences in the
main pipeline. They represent reviews where no explicit star rating was submitted.

The project keeps this decision explicit:

- `drop_zero_ratings()` removes zero-rated rows for the main explicit-feedback task.
- Tests verify zero-rated rows are removed without dropping nonzero rows.
- The notebook preserves both the data-quality discussion and the modeling decision.

Keeping zeros as ratings would create a false class of extreme negative feedback and
distort both RMSE and ranking metrics. Dropping them makes the target variable match
the semantics of explicit star prediction.

## 3. Rating Drift Is Real Enough to Model

Food.com interactions span roughly 18 years. The notebook analysis shows average
ratings changed over time, with a decline of about 0.4 rating points from the early
2000s into the mid-2010s before partial recovery.

That makes a static user-item model incomplete: the same user and recipe interaction
can sit in a different rating environment depending on the year. The implemented
`TimeAwareMF` adds user-time and item-time bias terms:

```text
r_hat(u, i, t) = global_mean + user_bias[u] + item_bias[i]
              + user_time_bias[u, t] + item_time_bias[i, t]
              + dot(user_factors[u], item_factors[i])
```

On the verified Phase 3 CLI run, time-aware MF reached:

| Metric | Value |
|--------|------:|
| Best validation RMSE | 0.6808 |
| Test RMSE | 0.6834 |
| NDCG@10 | 0.2252 |
| Recall@10 | 0.3153 |
| Ranking users evaluated | 50,089 |

These values are exported in `results/phase3_metrics.json`.

## 4. Cold-Start Remains the Main Failure Mode

The warm-start split still contains many recipes with limited training history. The
Phase 3 CLI run quantified this directly:

| Item bucket | Test rows | RMSE |
|-------------|----------:|-----:|
| Cold, fewer than 5 training interactions | 67,249 | 0.7195 |
| Medium, 5-19 training interactions | 45,498 | 0.6426 |
| Warm, at least 20 training interactions | 47,981 | 0.6688 |

The cold bucket is the hardest case. This is the technical justification for testing
hybrid MF + semantic embeddings: collaborative filtering is strongest when interaction
history exists, while recipe text embeddings can provide a content signal for new or
sparse recipes.

The full-data hybrid experiment was run after the Phase 3 baseline and did not improve
the system:

| Model | Test RMSE | NDCG@10 | Recall@10 |
|-------|----------:|--------:|----------:|
| Time-aware MF | 0.6834 | 0.2252 | 0.3153 |
| Hybrid MF + LLM embeddings | 0.7039 | 0.2180 | 0.3074 |

Hybrid also had worse cold-start RMSE (`0.7282` for cold items versus `0.7195` for
time-aware MF). The implemented hybrid remains useful as an experimental artifact and
future-work direction, but the production service should default to `time_aware_mf`.

## 5. Offline Experimentation Framework

The project uses an offline experimentation workflow rather than claiming a live A/B
test:

- RMSE for explicit rating prediction.
- NDCG@10 and Recall@10 with sampled negatives for ranking quality.
- Cold-start RMSE buckets by item interaction count.
- Bootstrap confidence intervals for paired model comparison.
- Paired t-test on squared errors for statistical validation.

This is the right framing for the current project stage. A live A/B test would require
real users, randomized traffic allocation, guardrail metrics, and online logging. The
repo instead implements the offline decision framework needed before a model is worth
shipping.

## 6. Productionization Result

Phase 3 converted the notebook workflow into a reproducible service layer:

- CLI training pipeline: `scripts/train.py`.
- CLI evaluation pipeline: `scripts/evaluate.py`.
- Standalone recipe embedding cache: `scripts/embed_recipes.py`.
- FastAPI inference service with `/health`, `/predict`, `/recommend`, `/similar`, `/metrics`, and `/demo`.
- Optional FAISS vector store for `/similar`, with brute-force fallback when FAISS is unavailable.
- Docker Compose service that mounts trained model artifacts.
- API tests with mocked models so CI does not need large artifacts.

Verified locally on April 27, 2026:

- `.venv/bin/python -m pytest tests/ -q`: 147 tests passed.
- `python scripts/train.py --model time_aware`: produced `models/time_aware_mf.joblib`
  and `models/user_rated.joblib`.
- `.venv/bin/python scripts/train.py --model hybrid`: produced `models/hybrid_mf.joblib`
  and `results/hybrid_metrics.json`, but metrics were worse than time-aware MF.
- FastAPI `/health`: HTTP 200 with `time_aware_mf` loaded.
- Docker Compose `/health`: HTTP 200 with `time_aware_mf` loaded.

The current service is a local/containerized ML inference API with a GitHub Actions
Cloud Run deploy path. The deployed Cloud Run revision returned `/health` after warmup
on April 27, 2026, but logs showed the old 2 GiB configuration exceeding memory during
live `/recommend` and `/similar` traffic. The deploy config now uses 4 GiB; push and
rerun the workflow before treating the public endpoint as fully production-ready.
