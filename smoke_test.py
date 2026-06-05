#!/usr/bin/env python3
"""Smoke test: load model and run quick predictions to verify it works.

Prints: model classes, prediction for a zero-row, prediction probabilities, and a basic sanity check.
"""
import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "engagement_recognition_model.joblib"
COLS_PATH = ROOT / "models" / "feature_columns.joblib"


def main():
    if not MODEL_PATH.exists():
        print(f"ERROR: model not found at {MODEL_PATH}")
        sys.exit(2)
    if not COLS_PATH.exists():
        print(f"ERROR: feature columns file not found at {COLS_PATH}")
        sys.exit(2)

    model = joblib.load(MODEL_PATH)
    cols = joblib.load(COLS_PATH)

    print("Model loaded OK.")
    print("Model classes:", getattr(model, 'classes_', 'N/A'))

    # create two test rows: zeros and random
    zero_row = {c: 0.0 for c in cols}
    rand_row = {c: float(np.random.normal(scale=1.0)) for c in cols}
    df = pd.DataFrame([zero_row, rand_row])

    try:
        preds = model.predict(df)
        print("Predictions:", preds)
    except Exception as e:
        print("Prediction failed:", e)
        sys.exit(3)

    if hasattr(model, 'predict_proba'):
        try:
            probs = model.predict_proba(df)
            print("Prediction probabilities shape:", probs.shape)
            print("First row probs sum (should be ~1):", probs[0].sum())
        except Exception as e:
            print("predict_proba failed:", e)
            sys.exit(4)

    print("Smoke test completed successfully.")

if __name__ == '__main__':
    main()
