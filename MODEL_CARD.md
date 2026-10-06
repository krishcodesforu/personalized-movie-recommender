# CineMatch model card

## Model and intended use

CineMatch is a hybrid movie recommender for portfolio demonstration and non-commercial experimentation. It combines item-item collaborative filtering, genre similarity, and a popularity fallback. A user creates a temporary profile by rating movies; the app ranks movies not present in that profile.

Do not use its predicted preference values as probabilities, quality judgments, or production decisions. Recommendations reflect historical patterns in MovieLens and may not match an individual's actual preferences.

## Training data and artifact

- **Dataset:** MovieLens Latest Small from [GroupLens Research](https://grouplens.org/datasets/movielens/latest/)
- **Dataset contents used:** 100,836 ratings across 9,742 movies, from 610 anonymized users
- **Training artifact:** [`artifacts/cinematch.joblib`](artifacts/cinematch.joblib)
- **Evaluation artifact:** [`artifacts/evaluation.json`](artifacts/evaluation.json)
- **Build command:** `cinematch build`

The serialized model includes a copy of rating interactions and movie data used for training. GroupLens allows redistribution of the dataset and transformations under its stated conditions. The data must be acknowledged, must not imply GroupLens endorsement, and must not be used for commercial or revenue-bearing purposes without prior permission. These restrictions apply to the serialized model, even though the CineMatch source code is MIT licensed. See the [upstream README and full terms](https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html).

## Evaluation

The evaluation uses seed 42 and holds out approximately 20% of each user's ratings. It fits a separate model on the remaining 80%, then recommends 10 movies for each user. Relevant items are held-out movies rated at least 4.0 stars. Recall@10 and NDCG@10 are macro-averaged across users with at least one relevant held-out item. Catalog coverage is the number of unique recommended movies divided by the full catalog size.

| Measure | Result |
| --- | ---: |
| Users evaluated | 604 |
| Training ratings | 80,672 |
| Held-out ratings | 20,164 |
| Relevant held-out items | 9,734 |
| Recall@10 | 0.0295 |
| NDCG@10 | 0.0537 |
| Catalog coverage@10 | 14.32% |

These are modest offline ranking results. No popularity-only baseline or temporal evaluation was run, so the results do not establish that the hybrid approach outperforms simpler alternatives or generalizes to future behavior. The dataset itself is a development dataset and is not suitable for shared research results.

## Reproducibility

The reported model was trained with Python 3.13.9, scikit-learn 1.7.2, NumPy 2.3.5, pandas 2.3.3, and joblib 1.5.2. Regenerate the saved artifact and metrics with:

```bash
cinematch build
cinematch evaluate
```

The evaluation split is deterministic for the recorded seed. Since the upstream Latest Small dataset may change, a future download may produce different metrics.

## Known limitations

- The ratings are historical and the Latest Small data was generated in 2018.
- User IDs are anonymized; demographic fairness and user-level impact cannot be assessed.
- Random per-user holdout can expose future interactions while predicting earlier ones; it is not a temporal recommendation test.
- Cold-start users need to rate at least two movies for personalized recommendations.
- The model has not been tested in a live product or with user studies.
