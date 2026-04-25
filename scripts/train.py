"""Training pipeline CLI.

Usage
-----
# Train all models (default)
python scripts/train.py --data-dir data/dataset --output-dir models

# Train only hybrid
python scripts/train.py --model hybrid --data-dir data/dataset --output-dir models

# Custom hyperparams
python scripts/train.py --model time_aware --k 10 --lr 0.005 --epochs 20
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import joblib

# allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import drop_zero_ratings, load_interactions, load_recipes, preprocess
from src.embeddings import RecipeEmbedder
from src.hybrid import HybridMF
from src.metrics import cold_start_rmse, rmse, sampled_evaluation
from src.models import StaticMF, TimeAwareMF
from src.splits import random_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")

_MODEL_CHOICES = ("static", "time_aware", "hybrid", "all")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train recipe recommender models")
    p.add_argument("--model", choices=_MODEL_CHOICES, default="all")
    p.add_argument("--data-dir", default="data/dataset",
                   help="Directory containing RAW_interactions.csv and RAW_recipes.csv")
    p.add_argument("--output-dir", default="models")
    p.add_argument("--k", type=int, default=None, help="MF latent factors (default: model-specific)")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--lamb", type=float, default=0.02)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--no-cache", action="store_true", help="Re-train even if cached model exists")
    p.add_argument("--results-file", default="results/metrics.json")
    return p.parse_args()


def load_data(data_dir: str):
    interactions = load_interactions(Path(data_dir) / "RAW_interactions.csv")
    recipes = load_recipes(Path(data_dir) / "RAW_recipes.csv")
    df = preprocess(interactions, recipes)
    df = drop_zero_ratings(df)
    return df, recipes


def evaluate_model(model, test_df, train_df, label: str) -> dict:
    preds = model.predict_batch(test_df)
    y_true = test_df["rating"].to_numpy(float)
    test_rmse = rmse(y_true, preds)

    cs = cold_start_rmse(model, test_df, train_df)
    rank = sampled_evaluation(model, test_df, train_df, n_negatives=100, k=10)

    result = {
        "model": label,
        "test_rmse": round(test_rmse, 4),
        "ndcg@10": round(rank["ndcg@10"], 4),
        "recall@10": round(rank["recall@10"], 4),
        "cold_start": {k: {"rmse": round(v["rmse"], 4), "n": v["n"]} for k, v in cs.items()},
    }
    logger.info(
        "%s | test_rmse=%.4f | ndcg@10=%.4f | recall@10=%.4f",
        label, test_rmse, rank["ndcg@10"], rank["recall@10"],
    )
    return result


def build_user_rated(train_df):
    """Return user_id -> set(recipe_id) from the training interactions."""
    user_rated = defaultdict(set)
    for user_id, recipe_id in zip(train_df["user_id"], train_df["recipe_id"]):
        user_rated[user_id].add(recipe_id)
    return dict(user_rated)


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    Path(args.results_file).parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data from %s …", args.data_dir)
    df, recipes = load_data(args.data_dir)
    split = random_split(df, seed=args.seed)
    logger.info("Split: train=%d  val=%d  test=%d", len(split.train), len(split.val), len(split.test))

    user_rated_path = out / "user_rated.joblib"
    joblib.dump(build_user_rated(split.train), user_rated_path)
    logger.info("Saved user-rated lookup to %s", user_rated_path)

    results = []
    train_models = (
        ["static", "time_aware", "hybrid"] if args.model == "all" else [args.model]
    )

    # ── Embedder (needed for hybrid) ─────────────────────────────────────────
    embedder = None
    if "hybrid" in train_models:
        embed_path = out / "recipe_embedder.joblib"
        if embed_path.exists() and not args.no_cache:
            logger.info("Loading cached embedder from %s", embed_path)
            embedder = RecipeEmbedder.load(embed_path)
        else:
            logger.info("Fitting RecipeEmbedder (%s) …", args.embed_model)
            recipe_cols = ["id", "name", "ingredients"]
            available = [c for c in recipe_cols if c in recipes.columns]
            embedder = RecipeEmbedder(model_name=args.embed_model)
            embedder.fit(recipes[available].drop_duplicates("id"))
            embedder.save(embed_path)

    # ── Static MF ────────────────────────────────────────────────────────────
    if "static" in train_models:
        path = out / "static_mf.joblib"
        if path.exists() and not args.no_cache:
            logger.info("Loading cached StaticMF")
            model = StaticMF.load(path)
        else:
            t0 = time.time()
            model = StaticMF(
                k=args.k or 10, lr=args.lr, lamb=args.lamb,
                epochs=args.epochs, patience=args.patience, seed=args.seed,
            )
            model.fit(split.train, split.val)
            model.save(path)
            logger.info("StaticMF trained in %.1fs", time.time() - t0)
        results.append(evaluate_model(model, split.test, split.train, "static_mf"))

    # ── Time-Aware MF ─────────────────────────────────────────────────────────
    if "time_aware" in train_models:
        path = out / "time_aware_mf.joblib"
        if path.exists() and not args.no_cache:
            logger.info("Loading cached TimeAwareMF")
            model = TimeAwareMF.load(path)
        else:
            t0 = time.time()
            model = TimeAwareMF(
                k=args.k or 5, lr=args.lr, lamb=args.lamb,
                epochs=args.epochs, patience=args.patience, seed=args.seed,
            )
            model.fit(split.train, split.val)
            model.save(path)
            logger.info("TimeAwareMF trained in %.1fs", time.time() - t0)
        results.append(evaluate_model(model, split.test, split.train, "time_aware_mf"))

    # ── Hybrid MF + LLM ───────────────────────────────────────────────────────
    if "hybrid" in train_models:
        path = out / "hybrid_mf.joblib"
        if path.exists() and not args.no_cache:
            logger.info("Loading cached HybridMF")
            model = HybridMF.load(path)
        else:
            t0 = time.time()
            model = HybridMF(
                mf=TimeAwareMF(
                    k=args.k or 5, lr=args.lr, lamb=args.lamb,
                    epochs=args.epochs, patience=args.patience, seed=args.seed,
                ),
                embedder=embedder,
            )
            model.fit(split.train, split.val)
            model.save(path)
            logger.info("HybridMF trained in %.1fs", time.time() - t0)
        results.append(evaluate_model(model, split.test, split.train, "hybrid_mf"))

    # ── Save results ─────────────────────────────────────────────────────────
    with open(args.results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results written to %s", args.results_file)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{'Model':<20} {'RMSE':>8} {'NDCG@10':>10} {'Recall@10':>12}")
    print("-" * 60)
    for r in results:
        print(f"{r['model']:<20} {r['test_rmse']:>8.4f} {r['ndcg@10']:>10.4f} {r['recall@10']:>12.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
