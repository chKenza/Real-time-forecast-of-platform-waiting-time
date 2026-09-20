"""
CLI entry point: preprocess data, train/tune a model, evaluate, and optionally
produce a test-set submission file.

Usage
-----
    python -m src.train --model random_forest --tune
    python -m src.train --model random_forest --sample-frac 0.1   # fast dev run
    python -m src.train --model random_forest --tune --predict-test

Run ``python -m src.train --help`` for the full list of options.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.model import MODEL_REGISTRY, evaluate, hyperparameter_search, train_model
from src.preprocessing import (
    TARGET_COLUMN,
    get_default_data_paths,
    load_csv,
    load_xy,
    preprocess_features,
)
from src.utils import save_metadata, save_model, time_based_split


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments for the training CLI.

    :param argv: argument list to parse; defaults to ``sys.argv[1:]``
    :return: parsed arguments
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Directory with the raw CSV files.")
    parser.add_argument(
        "--model",
        default="random_forest",
        choices=sorted(MODEL_REGISTRY),
        help="Which model from the registry to train.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run a grid search over the model's param_grid instead of using its defaults.",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.2,
        help="Fraction of the most recent unique dates held out for validation.",
    )
    parser.add_argument(
        "--clip-value",
        type=float,
        default=30.0,
        help="Absolute value to clip the past-delay features to.",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="If set, randomly subsample this fraction of training rows (fast dev iteration).",
    )
    parser.add_argument(
        "--output",
        default="models/model.joblib",
        help="Where to save the final fitted pipeline.",
    )
    parser.add_argument(
        "--predict-test",
        action="store_true",
        help="After training, predict on x_test and write a submission CSV.",
    )
    parser.add_argument(
        "--submission-output",
        default="outputs/submission.csv",
        help="Where to write the test-set submission CSV (with --predict-test).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """
    Run the train/tune/evaluate/submit pipeline described in the module docstring.

    :param argv: argument list to parse; defaults to ``sys.argv[1:]``
    """
    args = parse_args(argv)
    t_start = time.time()

    paths = get_default_data_paths(args.data_dir)
    print(f"Loading data from {args.data_dir} ...", flush=True)
    X_raw, y = load_xy(paths["x_train"], paths["y_train"])

    if args.sample_frac is not None:
        X_raw = X_raw.sample(frac=args.sample_frac, random_state=42)
        y = y.loc[X_raw.index]
        print(f"Subsampled to {len(X_raw)} rows (sample_frac={args.sample_frac})", flush=True)

    Xf = preprocess_features(X_raw, clip_value=args.clip_value)
    X_tr, X_val, y_tr, y_val = time_based_split(
        Xf, y, X_raw["date"], validation_size=args.validation_size
    )
    print(f"train={X_tr.shape}, validation={X_val.shape}", flush=True)

    baseline_mae = float(np.abs(y_val).mean())
    print(f"Naive baseline MAE (predict 0): {baseline_mae:.4f}", flush=True)

    if args.tune:
        print(f"Tuning '{args.model}' with grid search on the validation split ...", flush=True)
        t0 = time.time()
        pipeline, chosen_params, val_mae = hyperparameter_search(
            X_tr, y_tr, X_val, y_val, model_name=args.model
        )
        print(f"Search done in {time.time() - t0:.1f}s. Best params: {chosen_params}", flush=True)
    else:
        print(f"Training '{args.model}' with default params ...", flush=True)
        t0 = time.time()
        chosen_params = {}
        pipeline = train_model(X_tr, y_tr, model_name=args.model)
        val_mae = evaluate(pipeline, X_val, y_val)
        print(f"Fit done in {time.time() - t0:.1f}s.", flush=True)

    print(f"Validation MAE: {val_mae:.4f} (baseline: {baseline_mae:.4f})", flush=True)

    save_model(pipeline, args.output)
    print(f"Saved model to {args.output}", flush=True)

    metadata = {
        "model_name": args.model,
        "model_params": pipeline.named_steps["model"].get_params(),
        "tuned": args.tune,
        "validation_mae": val_mae,
        "baseline_mae": baseline_mae,
        "train_rows": len(X_tr),
        "validation_rows": len(X_val),
        "validation_size": args.validation_size,
        "clip_value": args.clip_value,
        "sample_frac": args.sample_frac,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = Path(args.output).with_suffix(".json")
    save_metadata(metadata, metadata_path)
    print(f"Saved metadata to {metadata_path}", flush=True)

    if args.predict_test:
        print("Refitting on train+validation for deployment ...", flush=True)
        production_pipeline = train_model(Xf, y, model_name=args.model, **chosen_params)
        production_output = Path(args.output).with_name(
            f"{Path(args.output).stem}_production{Path(args.output).suffix}"
        )
        save_model(production_pipeline, production_output)
        print(f"Saved production model to {production_output}", flush=True)

        print("Predicting on x_test ...", flush=True)
        x_test_raw = load_csv(paths["x_test"])
        Xt = preprocess_features(x_test_raw, clip_value=args.clip_value)
        preds = production_pipeline.predict(Xt)
        submission = pd.DataFrame({TARGET_COLUMN: preds})
        out_path = Path(args.submission_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(out_path)
        print(f"Wrote submission to {out_path}", flush=True)

    print(f"Total time: {time.time() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
