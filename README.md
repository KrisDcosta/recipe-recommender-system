# Recipe Recommender: Time-Aware Collaborative Filtering

Rating prediction on 1.1M Food.com interactions. Models progress from simple baselines
to time-aware matrix factorization that explicitly captures 18 years of rating drift.

## Results

Zero-rated interactions dropped: in Food.com, a rating of 0 means no explicit rating was given
(not a 1-star review). Including them inflates RMSE and pollutes the signal.

| Model | Test RMSE |
|-------|-----------|
| Recipe mean baseline | ~1.24 |
| Linear regression (recipe features) | ~1.18 |
| TF-IDF Ridge (ingredients) | ~1.17 |
| Static MF (SGD, k=10, λ=0.02) | ~0.73 |
| **Time-aware MF (SGD, k=5, λ=0.02)** | **0.69** |

Split: 70% train / 15% val / 15% test · Seed 42 · Zero ratings dropped (60,847 of 1,132,367)

## Key Findings

- **Rating drift is real**: average rating fell ~0.4 points from 2002→2014, then partially recovered — time bins capture signal static MF misses
- **Item-CF underperforms linear regression**: severe cold-start (only 38K recipe overlap between val and test sets) makes Pearson similarity unreliable
- **User profile TF-IDF is worse than baseline**: ingredient overlap alone doesn't predict taste — users rating similar recipes don't rate them similarly
- **Best model**: time-aware MF — user bias, item bias, user×time bias, item×time bias + latent factors trained with SGD + early stopping (RMSE 0.69)

## Dataset

Food.com Recipes and Interactions — [Kaggle](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)

```
data/dataset/
├── RAW_recipes.csv       # 231,637 recipes
└── RAW_interactions.csv  # 1,132,367 ratings
```

Download and place in `data/dataset/` before running.

## Setup

```bash
pip install -r requirements.txt
jupyter lab assignment2_1.ipynb
```

Run cells top-to-bottom. Data loading (~30s), Item-CF (~10min), MF training (~15min).

## Notebook Structure

| Section | Content |
|---------|---------|
| 1. Data Loading & Preprocessing | Load CSVs, nutrition parsing, merge, 70/15/15 split |
| EDA | Rating distributions, time trends, cold-start analysis |
| 2. Baseline Models | Recipe mean, Item-CF, linear regression, TF-IDF Ridge, static MF |
| 3. Time-Aware MF | SGD with user×time and item×time bias terms |
| 4. Results Comparison | RMSE table across all models |

## Course

CSE 258R: Recommender Systems & Web Mining · UC San Diego · Fall 2025
