"""Standalone recipe embedding script.

Run once to encode all recipes and cache the embedder.
The API and training pipeline load this cached file — no re-encoding needed.

Usage
-----
python scripts/embed_recipes.py --data-dir data/dataset --output models/recipe_embedder.joblib
python scripts/embed_recipes.py --data-dir data/dataset --model sentence-transformers/all-MiniLM-L6-v2
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_recipes
from src.embeddings import RecipeEmbedder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("embed_recipes")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Encode all recipes into dense embeddings")
    p.add_argument("--data-dir", default="data/dataset")
    p.add_argument("--output", default="models/recipe_embedder.joblib")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--force", action="store_true", help="Re-encode even if output exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.output)

    if out_path.exists() and not args.force:
        logger.info("Embedder already exists at %s. Use --force to re-encode.", out_path)
        return

    recipes = load_recipes(Path(args.data_dir) / "RAW_recipes.csv")
    recipe_cols = [c for c in ["id", "name", "ingredients"] if c in recipes.columns]
    recipes_embed = recipes[recipe_cols].drop_duplicates("id").reset_index(drop=True)
    logger.info("Encoding %d recipes with %s …", len(recipes_embed), args.model)

    embedder = RecipeEmbedder(model_name=args.model, batch_size=args.batch_size)
    embedder.fit(recipes_embed)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    embedder.save(out_path)
    logger.info("Done. Saved to %s", out_path)

    mat, ids = embedder.matrix()
    logger.info("Embedding matrix: %d recipes × %d dims", *mat.shape)


if __name__ == "__main__":
    main()
