"""Evaluate a saved model against the test split.

Usage
-----
python scripts/evaluate.py --model models/hybrid_mf.joblib --data-dir data/dataset
python scripts/evaluate.py --model models/time_aware_mf.joblib --output results/ta_eval.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import drop_zero_ratings, load_interactions, load_recipes, preprocess
from src.metrics import bootstrap_ci, cold_start_rmse, rmse, sampled_evaluation
from src.models import BaseMF
from src.splits import random_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger("evaluate")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a saved recommender model")
    p.add_argument("--model", required=True, help="Path to saved .joblib model")
    p.add_argument("--baseline", default=None, help="Optional second model for A/B comparison")
    p.add_argument("--data-dir", default="data/dataset")
    p.add_argument("--output", default=None, help="Write JSON results to this path")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=10, help="Ranking cutoff")
    p.add_argument("--n-negatives", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("Loading data …")
    interactions = load_interactions(Path(args.data_dir) / "RAW_interactions.csv")
    recipes = load_recipes(Path(args.data_dir) / "RAW_recipes.csv")
    df = drop_zero_ratings(preprocess(interactions, recipes))
    split = random_split(df, seed=args.seed)

    logger.info("Loading model from %s …", args.model)
    model = BaseMF.load(args.model)

    y_true = split.test["rating"].to_numpy(float)
    preds = model.predict_batch(split.test)
    test_rmse = rmse(y_true, preds)

    cs = cold_start_rmse(model, split.test, split.train)
    rank = sampled_evaluation(model, split.test, split.train,
                              n_negatives=args.n_negatives, k=args.k)

    result: dict = {
        "model_path": args.model,
        "test_rmse": round(test_rmse, 4),
        f"ndcg@{args.k}": round(rank[f"ndcg@{args.k}"], 4),
        f"recall@{args.k}": round(rank[f"recall@{args.k}"], 4),
        "n_users_evaluated": rank["n_users_evaluated"],
        "cold_start": {k: {"rmse": round(v["rmse"], 4), "n": v["n"]} for k, v in cs.items()},
    }

    # ── Optional A/B vs baseline ─────────────────────────────────────────────
    if args.baseline:
        logger.info("Loading baseline from %s …", args.baseline)
        baseline = BaseMF.load(args.baseline)
        preds_b = baseline.predict_batch(split.test)
        ab = bootstrap_ci(y_true, preds_b, preds, n_bootstrap=5_000, seed=args.seed)
        result["ab_vs_baseline"] = {
            "baseline_rmse": round(ab["rmse_a"], 4),
            "model_rmse": round(ab["rmse_b"], 4),
            "delta": round(ab["observed_delta"], 4),
            "p_value": round(ab["p_value"], 4),
            "significant": ab["significant"],
        }
        logger.info(
            "A/B: baseline=%.4f  model=%.4f  Δ=%.4f  p=%.4f  sig=%s",
            ab["rmse_a"], ab["rmse_b"], ab["observed_delta"], ab["p_value"], ab["significant"],
        )

    # ── Print ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"Model : {args.model}")
    print(f"RMSE  : {result['test_rmse']}")
    print(f"NDCG@{args.k}: {result[f'ndcg@{args.k}']}")
    print(f"Recall@{args.k}: {result[f'recall@{args.k}']}")
    print("\nCold-start breakdown:")
    for bucket, vals in result["cold_start"].items():
        print(f"  {bucket}: RMSE={vals['rmse']}  n={vals['n']}")
    print("=" * 50 + "\n")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Results written to %s", args.output)


if __name__ == "__main__":
    main()
