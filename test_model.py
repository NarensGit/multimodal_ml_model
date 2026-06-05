#!/usr/bin/env python3
"""Simple CLI to load saved model and predict from a features CSV.

Usage:
  python test_model.py --features path/to/features.csv --out results/predictions.csv

The features CSV must contain the same columns saved in `models/feature_columns.joblib`.
"""
import argparse
import os
import sys
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "engagement_recognition_model.joblib"
COLS_PATH = ROOT / "models" / "feature_columns.joblib"
# Output labels expected by the guide: lowercase textual labels
# 0 -> disengaged, 1 -> engaged
LABEL_MAP = {0: "disengaged", 1: "engaged"}

LABELS = None


def load_model():
    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}")
        sys.exit(1)
    if not COLS_PATH.exists():
        print(f"Feature columns file not found: {COLS_PATH}")
        sys.exit(1)
    model = joblib.load(MODEL_PATH)
    cols = joblib.load(COLS_PATH)
    return model, cols


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True, help="CSV file with feature rows (columns must match saved feature columns)")
    p.add_argument("--out", default="results/predictions.csv", help="Output CSV for predictions")
    args = p.parse_args()

    model, cols = load_model()

    df = pd.read_csv(args.features)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print("The features CSV is missing required columns:", missing)
        print("Available columns in the CSV:", list(df.columns)[:20])
        sys.exit(1)

    X = df[cols].copy()
    X = X.fillna(0)

    preds = model.predict(X)
    out_df = df.copy()
    out_df["predicted_label"] = preds
    out_df["predicted_label_name"] = out_df["predicted_label"].map(LABEL_MAP).fillna(out_df["predicted_label"].astype(str))

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        # add probability columns
        for i, cls in enumerate(model.classes_):
            out_df[f"prob_{cls}"] = probs[:, i]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote predictions to: {out_path}")


if __name__ == "__main__":
    main()
