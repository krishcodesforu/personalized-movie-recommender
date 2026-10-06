"""Hybrid item-based recommendation model."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer, normalize


class CineMatchRecommender:
    """Recommend movies from a small set of explicit profile ratings."""

    def __init__(self, movies: pd.DataFrame, ratings: pd.DataFrame, n_neighbors: int = 40):
        self.movies = movies.reset_index(drop=True).copy()
        self.ratings = ratings.copy()
        self.n_neighbors = n_neighbors
        self.movie_ids = self.movies["movieId"].astype(int).to_numpy()
        self.movie_to_index = {movie_id: index for index, movie_id in enumerate(self.movie_ids)}

        user_ids = self.ratings["userId"].astype(int).to_numpy()
        user_codes, self.user_ids = pd.factorize(user_ids, sort=True)
        item_codes = self.ratings["movieId"].astype(int).map(self.movie_to_index).to_numpy()
        if pd.isna(item_codes).any():
            raise ValueError("Ratings contain movie IDs absent from the movie catalog.")
        item_codes = item_codes.astype(int)
        user_means = self.ratings.groupby("userId")["rating"].mean()
        centered = (
            self.ratings["rating"].to_numpy(dtype=float)
            - self.ratings["userId"].map(user_means).to_numpy(dtype=float)
        )
        item_user = csr_matrix(
            (centered, (item_codes, user_codes)),
            shape=(len(self.movies), len(self.user_ids)),
            dtype=np.float32,
        )
        self.item_user = normalize(item_user, norm="l2", axis=1, copy=True).tocsr()
        self.neighbor_model = NearestNeighbors(
            metric="cosine", algorithm="brute", n_jobs=-1
        ).fit(self.item_user)

        genre_lists = self.movies["genres"].fillna("").map(
            lambda value: [genre for genre in str(value).split("|") if genre and genre != "(no genres listed)"]
        )
        self.genre_encoder = MultiLabelBinarizer(sparse_output=True)
        self.genre_matrix = normalize(
            self.genre_encoder.fit_transform(genre_lists).tocsr(), norm="l2", axis=1
        ).tocsr()

        self.global_mean = float(self.ratings["rating"].mean())
        aggregates = self.ratings.groupby("movieId")["rating"].agg(["mean", "count"])
        self.movie_mean = np.full(len(self.movies), self.global_mean, dtype=float)
        self.movie_count = np.zeros(len(self.movies), dtype=float)
        for movie_id, row in aggregates.iterrows():
            index = self.movie_to_index.get(int(movie_id))
            if index is not None:
                self.movie_mean[index] = float(row["mean"])
                self.movie_count[index] = float(row["count"])
        prior_weight = 20.0
        self.popularity_score = (
            self.movie_count / (self.movie_count + prior_weight) * self.movie_mean
            + prior_weight / (self.movie_count + prior_weight) * self.global_mean
        )

    def recommend(
        self,
        profile: Mapping[int, float],
        n: int = 10,
        strategy: str = "Hybrid",
    ) -> pd.DataFrame:
        """Rank unseen movies for {movie_id: rating} preferences."""
        strategy = strategy.lower().replace("-", " ")
        valid_profile = {
            int(movie_id): float(rating)
            for movie_id, rating in profile.items()
            if int(movie_id) in self.movie_to_index and 0.5 <= float(rating) <= 5.0
        }
        profile_ids = list(valid_profile)
        rated_indices = np.array([self.movie_to_index[movie_id] for movie_id in profile_ids], dtype=int)
        candidate_mask = np.ones(len(self.movies), dtype=bool)
        candidate_mask[rated_indices] = False

        if not valid_profile:
            predicted_cf = np.full(len(self.movies), self.global_mean, dtype=float)
            predicted_content = predicted_cf.copy()
            best_seed_for_item = np.full(len(self.movies), -1, dtype=int)
        else:
            profile_values = np.array([valid_profile[movie_id] for movie_id in profile_ids], dtype=float)
            profile_center = float(profile_values.mean())
            deviations = profile_values - profile_center
            cf_num = np.zeros(len(self.movies), dtype=float)
            cf_den = np.zeros(len(self.movies), dtype=float)
            best_seed_for_item = np.full(len(self.movies), -1, dtype=int)
            best_seed_similarity = np.full(len(self.movies), -1.0, dtype=float)

            neighbor_count = min(self.n_neighbors + 1, len(self.movies))
            distances, neighbors = self.neighbor_model.kneighbors(
                self.item_user[rated_indices], n_neighbors=neighbor_count
            )
            for seed_position, (row_distances, row_neighbors) in enumerate(zip(distances, neighbors)):
                for distance, neighbor_index in zip(row_distances, row_neighbors):
                    similarity = max(0.0, 1.0 - float(distance))
                    if neighbor_index in rated_indices or similarity <= 0.0:
                        continue
                    cf_num[neighbor_index] += similarity * deviations[seed_position]
                    cf_den[neighbor_index] += similarity
                    if similarity > best_seed_similarity[neighbor_index]:
                        best_seed_similarity[neighbor_index] = similarity
                        best_seed_for_item[neighbor_index] = seed_position
            predicted_cf = np.full(len(self.movies), self.global_mean, dtype=float)
            has_cf = cf_den > 0
            predicted_cf[has_cf] = profile_center + cf_num[has_cf] / cf_den[has_cf]

            content_similarities = (self.genre_matrix @ self.genre_matrix[rated_indices].T).toarray()
            content_num = content_similarities @ deviations
            content_den = content_similarities.sum(axis=1)
            predicted_content = np.full(len(self.movies), self.global_mean, dtype=float)
            has_content = content_den > 0
            predicted_content[has_content] = profile_center + content_num[has_content] / content_den[has_content]
            for item_index in range(len(self.movies)):
                if best_seed_for_item[item_index] < 0 and content_den[item_index] > 0:
                    best_seed_for_item[item_index] = int(np.argmax(content_similarities[item_index]))

        predicted_cf = np.clip(predicted_cf, 0.5, 5.0)
        predicted_content = np.clip(predicted_content, 0.5, 5.0)
        if strategy == "collaborative":
            scores = 0.9 * predicted_cf + 0.1 * self.popularity_score
        elif strategy == "content based" or strategy == "content":
            scores = 0.9 * predicted_content + 0.1 * self.popularity_score
        elif strategy == "popular":
            scores = self.popularity_score.copy()
        else:
            scores = 0.65 * predicted_cf + 0.25 * predicted_content + 0.10 * self.popularity_score

        candidates = np.flatnonzero(candidate_mask)
        ranked = candidates[np.argsort(scores[candidates])[::-1][: max(1, int(n))]]
        result = self.movies.iloc[ranked][["movieId", "title", "genres", "year"]].copy()
        result["predicted_rating"] = np.round(scores[ranked], 2)
        reasons: list[str] = []
        for item_index in ranked:
            seed_position = best_seed_for_item[item_index]
            if seed_position < 0 or not profile_ids:
                reasons.append("A strong pick in the MovieLens rating data")
            else:
                seed_id = profile_ids[seed_position]
                seed_title = self.movies.iloc[self.movie_to_index[seed_id]]["title"]
                reasons.append(f"Because you liked {seed_title}")
        result["reason"] = reasons
        return result.reset_index(drop=True)
