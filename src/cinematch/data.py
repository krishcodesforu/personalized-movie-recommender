"""Download and prepare MovieLens data."""

from __future__ import annotations

import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
ARCHIVE_NAME = "ml-latest-small.zip"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_paths(root: Path | None = None) -> dict[str, Path]:
    base = root or project_root()
    return {
        "archive": base / "data" / "raw" / ARCHIVE_NAME,
        "movies": base / "data" / "processed" / "movies.csv",
        "ratings": base / "data" / "processed" / "ratings.csv",
    }


def _download_archive(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        DATA_URL,
        headers={"User-Agent": "CineMatch-portfolio/1.0 (public MovieLens data)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        if not zipfile.is_zipfile(temporary):
            raise ValueError("The MovieLens download was not a valid ZIP archive.")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_dataset(root: Path | None = None, force_download: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return cleaned movies and ratings, downloading the archive on first use."""
    paths = data_paths(root)
    archive = paths["archive"]
    if force_download or not archive.exists() or not zipfile.is_zipfile(archive):
        _download_archive(archive)

    if not force_download and paths["movies"].exists() and paths["ratings"].exists():
        return _read_processed(paths["movies"], paths["ratings"])

    paths["movies"].parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        movie_file = "ml-latest-small/movies.csv"
        rating_file = "ml-latest-small/ratings.csv"
        if movie_file not in names or rating_file not in names:
            raise ValueError("The MovieLens archive is missing movies.csv or ratings.csv.")
        with bundle.open(movie_file) as file:
            movies = pd.read_csv(file)
        with bundle.open(rating_file) as file:
            ratings = pd.read_csv(file)

    movies, ratings = _clean_data(movies, ratings)
    movies.to_csv(paths["movies"], index=False)
    ratings.to_csv(paths["ratings"], index=False)
    return movies, ratings


def _read_processed(movie_path: Path, rating_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    movies = pd.read_csv(movie_path)
    ratings = pd.read_csv(rating_path)
    return _clean_data(movies, ratings)


def _clean_data(movies: pd.DataFrame, ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_movies = {"movieId", "title", "genres"}
    required_ratings = {"userId", "movieId", "rating"}
    if not required_movies.issubset(movies.columns):
        raise ValueError(f"Movie data must contain: {', '.join(sorted(required_movies))}")
    if not required_ratings.issubset(ratings.columns):
        raise ValueError(f"Rating data must contain: {', '.join(sorted(required_ratings))}")

    clean_movies = movies[["movieId", "title", "genres"]].copy()
    clean_movies["movieId"] = pd.to_numeric(clean_movies["movieId"], errors="coerce")
    clean_movies = clean_movies.dropna(subset=["movieId", "title"]).drop_duplicates("movieId")
    clean_movies["movieId"] = clean_movies["movieId"].astype(int)
    clean_movies["title"] = clean_movies["title"].astype(str).str.strip()
    clean_movies["genres"] = clean_movies["genres"].fillna("(no genres listed)").astype(str)
    clean_movies["year"] = pd.to_numeric(
        clean_movies["title"].str.extract(r"\((\d{4})\)\s*$")[0], errors="coerce"
    ).astype("Int64")

    clean_ratings = ratings[["userId", "movieId", "rating"]].copy()
    for column in ("userId", "movieId", "rating"):
        clean_ratings[column] = pd.to_numeric(clean_ratings[column], errors="coerce")
    clean_ratings = clean_ratings.dropna(subset=["userId", "movieId", "rating"])
    clean_ratings["userId"] = clean_ratings["userId"].astype(int)
    clean_ratings["movieId"] = clean_ratings["movieId"].astype(int)
    clean_ratings = clean_ratings[clean_ratings["rating"].between(0.5, 5.0)]
    clean_ratings = clean_ratings[clean_ratings["movieId"].isin(clean_movies["movieId"])]
    clean_ratings = clean_ratings.drop_duplicates(["userId", "movieId"], keep="last")
    return clean_movies.reset_index(drop=True), clean_ratings.reset_index(drop=True)
