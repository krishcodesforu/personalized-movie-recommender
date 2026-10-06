"""Build the recommender artifact and calculate offline ranking metrics."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from cinematch.data import load_dataset, project_root
from cinematch.model import CineMatchRecommender


def artifact_path(root: Path | None = None) -> Path:
    return (root or project_root()) / "artifacts" / "cinematch.joblib"


def build_model(root: Path | None = None, force_download: bool = False) -> CineMatchRecommender:
    """Load public data, train the model, and save a local artifact."""
    base = root or project_root()
    movies, ratings = load_dataset(root=base, force_download=force_download)
    model = CineMatchRecommender(movies, ratings)
    destination = artifact_path(base)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        joblib.dump(model, temporary, compress=3)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return model


def evaluate_model(root: Path | None = None, seed: int = 42, k: int = 10) -> dict[str, float | int]:
    """Perform deterministic per-user holdout evaluation on the public ratings."""
    base = root or project_root()
    movies, ratings = load_dataset(root=base)
    rng = np.random.default_rng(seed)
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for _, group in ratings.groupby("userId", sort=True):
        if len(group) < 5:
            train_parts.append(group)
            continue
        indices = rng.permutation(group.index.to_numpy())
        test_size = max(1, int(round(len(group) * 0.2)))
        test_indices = indices[:test_size]
        test_parts.append(group.loc[test_indices])
        train_parts.append(group.loc[~group.index.isin(test_indices)])

    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True) if test_parts else ratings.iloc[0:0].copy()
    model = CineMatchRecommender(movies, train)
    recalls: list[float] = []
    ndcgs: list[float] = []
    recommended_items: set[int] = set()
    evaluated_users = 0
    relevant_count = 0

    training_by_user = {int(user_id): group for user_id, group in train.groupby("userId")}
    for user_id, held_out in test.groupby("userId"):
        relevant = set(held_out.loc[held_out["rating"] >= 4.0, "movieId"].astype(int))
        if not relevant:
            continue
        user_history = training_by_user.get(int(user_id))
        if user_history is None or user_history.empty:
            continue
        profile_rows = user_history.sort_values(["rating", "movieId"], ascending=[False, True]).head(20)
        profile = dict(zip(profile_rows["movieId"].astype(int), profile_rows["rating"].astype(float)))
        recommendations = model.recommend(profile, n=k, strategy="Hybrid")
        ranked = recommendations["movieId"].astype(int).tolist()
        recommended_items.update(ranked)
        hits = [1 if movie_id in relevant else 0 for movie_id in ranked]
        recalls.append(sum(hits) / len(relevant))
        dcg = sum(hit / np.log2(position + 2) for position, hit in enumerate(hits))
        ideal_hits = min(len(relevant), k)
        ideal_dcg = sum(1.0 / np.log2(position + 2) for position in range(ideal_hits))
        ndcgs.append(dcg / ideal_dcg if ideal_dcg else 0.0)
        relevant_count += len(relevant)
        evaluated_users += 1

    metrics: dict[str, float | int] = {
        "seed": int(seed),
        "k": int(k),
        "users_evaluated": int(evaluated_users),
        "relevant_holdout_items": int(relevant_count),
        f"recall_at_{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg_at_{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "catalog_coverage_at_k": float(len(recommended_items) / max(1, len(movies))),
        "train_ratings": int(len(train)),
        "test_ratings": int(len(test)),
    }
    output = base / "artifacts" / "evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics
