#!/usr/bin/env python3
"""Retrain the engagement model as a binary classifier.

This script uses the balanced binary dataset in `data/` and saves the trained
model back to `models/engagement_recognition_model.joblib` so the existing
prediction scripts and Flask UI can use the new model without changes.

Input dataset columns expected:
  - engaged: binary target (0 = Not Engaged, 1 = Engaged)
  - feature columns produced by preprocessing

Usage:
  python train_binary_model.py --data data/engagement_binary_dataset.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from create_processed_dataset import process_participant


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "engagement_recognition_model.joblib"
FEATURE_COLS_PATH = ROOT / "models" / "feature_columns.joblib"
METRICS_PATH = ROOT / "results" / "model_metrics.joblib"


def load_training_dataframe(data_path: Path) -> pd.DataFrame:
    """Load either a preprocessed binary CSV or build one from raw dataset folders."""
    if data_path.is_file():
        df = pd.read_csv(data_path)
        if "engaged" in df.columns:
            return df
        raise ValueError("CSV input must contain an 'engaged' target column.")

    if not data_path.is_dir():
        raise FileNotFoundError(f"Dataset path not found: {data_path}")

    rows = []
    for condition_dir in sorted([p for p in data_path.iterdir() if p.is_dir()]):
        for participant_dir in sorted([p for p in condition_dir.iterdir() if p.is_dir()]):
            participant_rows = process_participant(participant_dir)
            for row in participant_rows:
                row["condition"] = condition_dir.name
                row["engaged"] = 1 if "hpe" in condition_dir.name.lower() else 0
                row["engaged_label"] = "Engaged" if row["engaged"] == 1 else "Not Engaged"
                rows.append(row)

    if not rows:
        raise ValueError(f"No training rows could be built from {data_path}")

    return pd.DataFrame(rows)


def train_binary_model(data_path: Path, random_state: int = 42):
    df = load_training_dataframe(data_path)

    if "engaged" not in df.columns:
        raise ValueError("The dataset must include an 'engaged' binary target column.")

    # Drop non-feature columns.
    drop_cols = [c for c in ["participant", "condition", "start_time", "engaged_label", "label"] if c in df.columns]
    feature_cols = [c for c in df.columns if c not in drop_cols + ["engaged"]]

    X = df[feature_cols].copy()
    y = df["engaged"].astype(int)

    # Preserve class balance in the split.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Not Engaged", "Engaged"], output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_count": int(len(feature_cols)),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_cols, FEATURE_COLS_PATH)
    joblib.dump(metrics, METRICS_PATH)

    return model, feature_cols, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/Engagnition Dataset",
        help="Raw dataset folder or a preprocessed CSV with an 'engaged' column",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    model, feature_cols, metrics = train_binary_model(data_path, random_state=args.seed)

    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved feature columns to: {FEATURE_COLS_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")
    print(f"Features used: {len(feature_cols)}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print("Confusion matrix:")
    print(metrics["confusion_matrix"])
    print("Classification report:")
    print(pd.DataFrame(metrics["classification_report"]).T.to_string())


if __name__ == "__main__":
    main()
