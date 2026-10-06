# CineMatch — personalized movie recommendations

An end-to-end data science portfolio project that turns public movie ratings into an interactive recommender. Build a profile from movies you have seen, rate them, and get recommendations with short explanations.

## What it demonstrates

- Automated public dataset download and validation
- Data cleaning and feature preparation
- A hybrid recommender that combines item-item collaborative filtering, genre similarity, and a popularity fallback
- A reproducible per-user offline ranking evaluation with Recall@10, NDCG@10, and catalog coverage
- An interactive Streamlit app with profile import/export and downloadable recommendations
- Reproducible command-line workflows and a deployment-friendly setup

## Quick start

Requires Python 3.10 or newer.

```bash
git clone <your-repository-url>
cd cinematch
python -m venv .venv
```

Activate the virtual environment (PowerShell on Windows):

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux, use `source .venv/bin/activate`. Then install and launch:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
streamlit run app.py
```

The repository includes the trained model and its evaluation report. On first launch, CineMatch loads that model; if you remove it, the app downloads MovieLens Latest Small, prepares the data, and trains a replacement. To rebuild explicitly:

```bash
cinematch build
```

Run the reproducible offline evaluation:

```bash
cinematch evaluate
```

To run the same app in a container:

```bash
docker build -t cinematch .
docker run --rm -p 8501:8501 cinematch
```

The trained model and aggregate evaluation report are tracked in `artifacts/`. Downloaded raw and processed data stay out of Git. The serialized model contains the MovieLens rating interactions used to train it, so the dataset's terms apply to that artifact; review [MODEL_CARD.md](MODEL_CARD.md) before reusing it. Use `cinematch --help` for command options. A sample JSON profile is available at [`examples/sample_profile.json`](examples/sample_profile.json).

## Using the app

1. Search for movies you have watched.
2. Add several titles and rate each from 0.5 to 5 stars. The initial ratings can be edited in the table.
3. Choose Hybrid, Collaborative, Content-based, or Popular recommendations.
4. Review the predicted preference score and the “because you liked” explanation. Export recommendations as CSV or save your profile as JSON for a later session.

Profiles are kept in the browser session unless you download them. The app does not require an account. In a hosted deployment, ratings entered in the app are sent to the server running the app to calculate recommendations; the project does not save those profile ratings.

## How it works

The collaborative model represents each movie by its user-rating vector, centers ratings by user mean, and finds similar movies with cosine distance. A genre-based content model compares the genres of rated movies. Candidate scores blend collaborative predictions, content affinity, and a Bayesian-smoothed popularity estimate. If a movie has too little interaction history, content and popularity provide graceful fallbacks.

The evaluator holds out a deterministic sample of each user's ratings, fits on the remaining interactions, and measures whether highly rated held-out movies appear near the top of recommendations. This is a learning/demo evaluation, not a claim of production performance. The split is not a temporal test and MovieLens users are anonymized benchmark users. The current measured results and their scope are documented in [MODEL_CARD.md](MODEL_CARD.md).

## Data source and attribution

The pipeline downloads **MovieLens Latest Small** from [GroupLens Research](https://grouplens.org/datasets/movielens/latest/) using the stable archive URL `https://files.grouplens.org/datasets/movielens/ml-latest-small.zip`. The dataset contains movie metadata, ratings, tags, and external IDs; CineMatch uses the movie and rating files. GroupLens describes this as a development dataset that may change and is not appropriate for shared research results. It requires attribution, carries no endorsement, and may not be used for commercial or revenue-bearing purposes without prior permission. Read the included upstream README before redistribution or other use.

Suggested academic citation:

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems (TiiS)* 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872

The CineMatch code is MIT licensed; that license does not replace or extend the dataset's terms. Raw downloads are kept out of Git. The trained artifact is committed for convenient app startup; it contains rating interactions derived from MovieLens and is subject to the same upstream terms. The code's MIT license does not cover the trained artifact or dataset.

## Repository layout

```text
app.py                   Streamlit application
src/cinematch/data.py    Download, validate, and prepare the data
src/cinematch/model.py   Hybrid recommender and profile recommendations
src/cinematch/pipeline.py Build artifacts and evaluate ranking quality
src/cinematch/cli.py     Command-line entry point
data/                    Local raw and processed data (ignored by Git)
artifacts/               Trained model and aggregate evaluation report
```

## Share it on LinkedIn

Suggested post (replace the bracketed results after running `cinematch evaluate`):

> I built CineMatch, an end-to-end personalized movie recommendation system. It combines collaborative filtering, genre similarity, and popularity in a web app where users can rate movies and inspect explainable picks. On a deterministic per-user holdout, it achieved Recall@10 0.0295 and NDCG@10 0.0537, with 14.32% catalog coverage. The results are a baseline for future tuning, not a claim of production performance. Code and model card: https://github.com/krishcodesforu/personalized-movie-recommender

## Limitations and next steps

MovieLens is an older, anonymized dataset with limited metadata. Recommendations reflect patterns in those ratings, not current popularity or quality. Cold-start users need to rate a few titles, scores are ranking aids rather than calibrated probabilities, and offline metrics do not guarantee a better viewing experience. Possible extensions include time-aware evaluation, richer movie metadata, implicit-feedback methods, and deployment with persistent user profiles and appropriate privacy controls.
