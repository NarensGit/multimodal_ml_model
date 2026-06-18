#!/usr/bin/env python3
"""Train a 3-class engagement model from the raw Engagnition dataset.

Classes:
  0 = Baseline
  1 = Low Engagement (LPE)
  2 = High Engagement (HPE)

The trainer can be run on the raw dataset root or on a processed CSV created
by `create_processed_dataset.py`. It also supports a few feature combinations so
you can compare which signals are most useful.

Usage:
  python train_three_class_model.py --data "data/Engagnition Dataset" --feature-set all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from create_processed_dataset import infer_three_class_label, label_name, process_participant


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "engagement_recognition_model.joblib"
FEATURE_COLS_PATH = ROOT / "models" / "feature_columns.joblib"
METRICS_PATH = ROOT / "results" / "model_metrics.joblib"
FEATURE_IMPORTANCE_PATH = ROOT / "results" / "feature_importance_3class.csv"

META_COLUMNS = {
    "participant",
    "condition",
    "start_time",
    "label",
    "label_name",
    "engaged",
    "engaged_label",
}

FEATURE_SETS = {
    "sensor_only": ["GSR_", "Temperature_", "Acc_X_", "Acc_Y_", "Acc_Z_", "Acc_SVM_"],
    "sensor_plus_gaze": ["GSR_", "Temperature_", "Gaze_", "Acc_X_", "Acc_Y_", "Acc_Z_", "Acc_SVM_"],
    "sensor_plus_context": [
        "GSR_",
        "Temperature_",
        "Gaze_",
        "Acc_X_",
        "Acc_Y_",
        "Acc_Z_",
        "Acc_SVM_",
        "gaze_",
        "session_",
        "reported_time_spent_seconds",
        "intervention_",
    ],
    "all": None,
}


def load_training_dataframe(data_path: Path) -> pd.DataFrame:
    """Load a processed CSV or build a new 3-class dataframe from raw folders."""
    if data_path.is_file():
        df = pd.read_csv(data_path)
        if "label" in df.columns:
            return df
        if "condition" in df.columns:
            df = df.copy()
            df["label"] = df["condition"].apply(infer_three_class_label)
            df["label_name"] = df["label"].apply(label_name)
            return df
        raise ValueError("CSV input must contain either a 'label' or 'condition' column.")

    if not data_path.is_dir():
        raise FileNotFoundError(f"Dataset path not found: {data_path}")

    rows = []
    for condition_dir in sorted([path for path in data_path.iterdir() if path.is_dir()]):
        for participant_dir in sorted([path for path in condition_dir.iterdir() if path.is_dir()]):
            participant_rows = process_participant(participant_dir)
            for row in participant_rows:
                row["condition"] = condition_dir.name
                row["label"] = infer_three_class_label(condition_dir.name)
                row["label_name"] = label_name(row["label"])
                rows.append(row)

    if not rows:
        raise ValueError(f"No training rows could be built from {data_path}")

    return pd.DataFrame(rows)


def select_feature_columns(df: pd.DataFrame, feature_set: str) -> list[str]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}. Choose from {sorted(FEATURE_SETS)}")

    prefixes = FEATURE_SETS[feature_set]
    candidate_columns = [column for column in df.columns if column not in META_COLUMNS and column != "label"]

    if prefixes is None:
        return candidate_columns

    selected = []
    for column in candidate_columns:
        if any(column.startswith(prefix) for prefix in prefixes):
            selected.append(column)
    return selected


def train_three_class_model(data_path: Path, feature_set: str = "all", random_state: int = 42):
    df = load_training_dataframe(data_path)

    if "label" not in df.columns:
        raise ValueError("The dataset must include a 3-class 'label' column.")

    feature_cols = select_feature_columns(df, feature_set)
    if not feature_cols:
        raise ValueError(f"No feature columns matched feature set '{feature_set}'.")

    X = df[feature_cols].copy().fillna(0)
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=18,
        min_samples_split=8,
        min_samples_leaf=3,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=["Baseline", "Low Engagement", "High Engagement"],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    metrics = {
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_count": int(len(feature_cols)),
        "feature_set": feature_set,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_cols, FEATURE_COLS_PATH)
    joblib.dump(metrics, METRICS_PATH)

    feature_importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    return model, feature_cols, metrics, feature_importance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/Engagnition Dataset",
        help="Raw dataset folder or a processed CSV with a 'label' column",
    )
    parser.add_argument("--feature-set", default="all", choices=sorted(FEATURE_SETS), help="Feature combination to train on")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    model, feature_cols, metrics, feature_importance = train_three_class_model(data_path, feature_set=args.feature_set, random_state=args.seed)

    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved feature columns to: {FEATURE_COLS_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")
    print(f"Saved feature importance to: {FEATURE_IMPORTANCE_PATH}")
    print(f"Feature set: {args.feature_set}")
    print(f"Features used: {len(feature_cols)}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Weighted F1: {metrics['f1_weighted']:.4f}")
    print("Confusion matrix:")
    print(pd.DataFrame(metrics["confusion_matrix"], index=["Baseline", "Low", "High"], columns=["Baseline", "Low", "High"]))
    print("Classification report:")
    print(pd.DataFrame(metrics["classification_report"]).T.to_string())
    print("Top 10 features:")
    print(feature_importance.head(10).to_string(index=False))


if __name__ == "__main__":
    main()