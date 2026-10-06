"""CineMatch command-line interface."""

from __future__ import annotations

import argparse
import json

from cinematch.pipeline import build_model, evaluate_model


def main() -> None:
    parser = argparse.ArgumentParser(prog="cinematch", description="Build and evaluate the CineMatch recommender.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build", help="Download data if needed and train the recommender")
    build_parser.add_argument("--refresh-data", action="store_true", help="Download a fresh copy of the public dataset")
    evaluate_parser = subparsers.add_parser("evaluate", help="Run deterministic per-user top-k evaluation")
    evaluate_parser.add_argument("--seed", type=int, default=42)
    evaluate_parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    if args.command == "build":
        model = build_model(force_download=args.refresh_data)
        print(f"Built CineMatch for {len(model.movies):,} movies and {len(model.ratings):,} ratings.")
    elif args.command == "evaluate":
        print(json.dumps(evaluate_model(seed=args.seed, k=args.k), indent=2))
