#!/usr/bin/env python3
"""Create processed features from the raw Engagnition dataset.

This version keeps the original E4 sliding-window features and adds optional
gaze, session elapsed time, and intervention summary features when those files
are present. The output is suitable for the 3-class baseline / LPE / HPE model.

Usage:
  python create_processed_dataset.py --data-root "data/Engagnition Dataset" --out data/processed_features.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


ROOT = Path(__file__).resolve().parent


def normalize_condition_name(condition_name: object) -> str:
    condition = str(condition_name).strip().lower()
    if "baseline" in condition:
        return "Baseline condition"
    if "lpe" in condition or "low" in condition:
        return "LPE condition"
    if "hpe" in condition or "high" in condition:
        return "HPE condition"
    return str(condition_name).strip()


def infer_three_class_label(condition_name: object) -> int:
    condition = str(condition_name).strip().lower()
    if "baseline" in condition:
        return 0
    if "lpe" in condition or "low" in condition:
        return 1
    if "hpe" in condition or "high" in condition:
        return 2
    raise ValueError(f"Unknown condition value: {condition_name}")


def label_name(value: int) -> str:
    return {0: "baseline", 1: "low", 2: "high"}.get(int(value), str(value))


def to_seconds_series(time_series: pd.Series) -> pd.Series:
    values = pd.to_numeric(time_series, errors="coerce")
    non_null = values.dropna()
    if non_null.empty:
        return pd.Series(dtype=float)

    median = float(non_null.median())
    if median > 1e12:
        scale = 1e6
    elif median > 1e11:
        scale = 1e3
    else:
        scale = 1.0

    normalized = values / scale
    first_valid = normalized.dropna().iloc[0]
    return normalized - first_valid


def resample_to_1hz(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    if time_col not in df.columns:
        return pd.DataFrame()

    working = df.dropna(subset=[time_col]).copy()
    if working.empty:
        return working

    working["_relative_time"] = to_seconds_series(working[time_col])
    working = working.dropna(subset=["_relative_time"])
    if working.empty:
        return working

    working = working.sort_values("_relative_time")
    if working["_relative_time"].duplicated().any():
        working = working[~working["_relative_time"].duplicated(keep="first")]

    working = working.set_index("_relative_time")
    start = float(np.ceil(working.index.min()))
    end = float(np.floor(working.index.max()))
    if start >= end:
        return working.resample("1s").mean(numeric_only=True).interpolate()

    new_index = np.arange(start, end + 1.0, 1.0)
    working = working.reindex(working.index.union(new_index))
    working = working.interpolate(method="index").reindex(new_index)
    working.index.name = "relative_time"
    return working


def extract_basic_features(arr: np.ndarray) -> dict:
    if len(arr) == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "var": np.nan,
            "max": np.nan,
            "min": np.nan,
            "peak_count": 0,
            "motion_energy": 0.0,
        }

    values = np.asarray(arr, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "var": np.nan,
            "max": np.nan,
            "min": np.nan,
            "peak_count": 0,
            "motion_energy": 0.0,
        }

    try:
        peak_count = int(len(find_peaks(finite)[0]))
    except Exception:
        peak_count = 0

    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "var": float(np.var(finite)),
        "max": float(np.max(finite)),
        "min": float(np.min(finite)),
        "peak_count": peak_count,
        "motion_energy": float(np.sum(np.diff(finite) ** 2)) if len(finite) >= 2 else 0.0,
    }


def extract_sequence_summary(arr: np.ndarray, prefix: str) -> dict:
    values = np.asarray(arr, dtype=float)
    clean = values[~np.isnan(values)]
    if clean.size == 0:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_active_ratio": np.nan,
            f"{prefix}_transition_count": 0,
        }

    transitions = int(np.sum(np.diff(clean) != 0)) if len(clean) >= 2 else 0
    return {
        f"{prefix}_count": int(len(clean)),
        f"{prefix}_active_ratio": float(np.mean(clean)),
        f"{prefix}_transition_count": transitions,
    }


def window_features(df_1hz: pd.DataFrame, window_size: int = 60, step: int = 30):
    n_rows = len(df_1hz)
    results = []
    if n_rows < window_size:
        return results

    for start in range(0, n_rows - window_size + 1, step):
        stop = start + window_size
        window = df_1hz.iloc[start:stop]
        features = {}

        for signal_name in ["GSR", "Temperature", "Gaze", "Acc_X", "Acc_Y", "Acc_Z", "Acc_SVM"]:
            values = window[signal_name].values if signal_name in window.columns else np.array([])
            features.update({f"{signal_name}_{key}": value for key, value in extract_basic_features(values).items()})

        results.append((float(window.index[0]), features))

    return results


def find_numeric_col(df: pd.DataFrame, exclude=None):
    exclude = exclude or []
    for column in df.columns:
        if column in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            return column
    return None


def load_session_metadata(root: Path) -> dict:
    lookup = {}
    session_path = root / "Session Elapsed Time.xlsx"
    if not session_path.exists():
        return lookup

    try:
        raw = pd.read_excel(session_path, header=None)
    except Exception as exc:
        print(f"Failed to read {session_path.name}: {exc}")
        return lookup

    if raw.shape[0] < 3:
        return lookup

    data = raw.iloc[2:].copy()
    data = data.rename(columns={0: "condition", 1: "participant", 2: "reported_time_spent_seconds"})
    data = data.dropna(how="all")
    data["condition"] = data["condition"].ffill().apply(normalize_condition_name)
    data["participant"] = data["participant"].ffill().astype(str).str.strip()

    for _, row in data.iterrows():
        participant = str(row["participant"]).strip()
        condition = str(row["condition"]).strip()
        if not participant or participant.lower() == "nan":
            continue

        session_values = pd.to_numeric(row.iloc[3:], errors="coerce")
        valid_sessions = session_values.dropna()
        lookup[(condition, participant)] = {
            "reported_time_spent_seconds": float(pd.to_numeric(row["reported_time_spent_seconds"], errors="coerce")) if pd.notna(row["reported_time_spent_seconds"]) else np.nan,
            "session_elapsed_total_seconds": float(valid_sessions.sum()) if not valid_sessions.empty else np.nan,
            "session_elapsed_mean_seconds": float(valid_sessions.mean()) if not valid_sessions.empty else np.nan,
            "session_elapsed_std_seconds": float(valid_sessions.std()) if len(valid_sessions) >= 2 else 0.0,
            "session_elapsed_max_seconds": float(valid_sessions.max()) if not valid_sessions.empty else np.nan,
            "session_elapsed_min_seconds": float(valid_sessions.min()) if not valid_sessions.empty else np.nan,
            "session_elapsed_count": int(valid_sessions.count()),
            "session_elapsed_nonzero_count": int((valid_sessions > 0).sum()) if not valid_sessions.empty else 0,
        }

    return lookup


def load_intervention_metadata(root: Path) -> dict:
    lookup = {}
    intervention_path = root / "InterventionData.xlsx"
    if not intervention_path.exists():
        return lookup

    try:
        raw = pd.read_excel(intervention_path, header=None)
    except Exception as exc:
        print(f"Failed to read {intervention_path.name}: {exc}")
        return lookup

    if raw.shape[0] < 3:
        return lookup

    data = raw.iloc[2:].copy()
    data = data.rename(columns={1: "participant", 2: "condition", 3: "intervention_type", 4: "intervention_time_seconds"})
    data = data.dropna(how="all")
    data["participant"] = data["participant"].ffill().astype(str).str.strip()
    data["condition"] = data["condition"].ffill().apply(normalize_condition_name)
    data["intervention_type"] = data["intervention_type"].astype(str).str.strip()
    data["intervention_time_seconds"] = pd.to_numeric(data["intervention_time_seconds"], errors="coerce")

    for (condition, participant), group in data.groupby(["condition", "participant"]):
        valid_times = group["intervention_time_seconds"].dropna()
        lookup[(condition, participant)] = {
            "intervention_row_count": int(len(group)),
            "intervention_numeric_time_count": int(valid_times.count()),
            "intervention_numeric_time_mean": float(valid_times.mean()) if not valid_times.empty else np.nan,
            "intervention_numeric_time_std": float(valid_times.std()) if len(valid_times) >= 2 else 0.0,
            "intervention_numeric_time_min": float(valid_times.min()) if not valid_times.empty else np.nan,
            "intervention_numeric_time_max": float(valid_times.max()) if not valid_times.empty else np.nan,
            "intervention_discrete_count": int(group["intervention_type"].str.contains("discrete", case=False, na=False).sum()),
            "intervention_no_need_count": int(group["intervention_type"].str.contains("no need", case=False, na=False).sum()),
        }

    return lookup


def load_participant_data(participant_path: Path):
    """Load core signals plus optional gaze for one participant."""
    data = {}

    try:
        for key, filename in {
            "gsr": "E4GsrData.csv",
            "acc": "E4AccData.csv",
            "tmp": "E4TmpData.csv",
            "gaze": "GazeData.csv",
        }.items():
            file_path = participant_path / filename
            data[key] = pd.read_csv(file_path) if file_path.exists() else None
    except Exception as exc:
        print(f"Error loading data from {participant_path}: {exc}")
        return None

    return data if any(value is not None for value in data.values()) else None


def build_context_features(participant_data: dict, root: Path, condition_name: str, participant_name: str) -> dict:
    context = {}
    session_lookup = load_session_metadata(root)
    intervention_lookup = load_intervention_metadata(root)

    context.update(session_lookup.get((condition_name, participant_name), {}))
    context.update(intervention_lookup.get((condition_name, participant_name), {}))

    gaze_df = participant_data.get("gaze")
    if gaze_df is not None and not gaze_df.empty and "Gaze" in gaze_df.columns:
        gaze_values = pd.to_numeric(gaze_df["Gaze"], errors="coerce").values
        context.update({f"gaze_{key}": value for key, value in extract_basic_features(gaze_values).items()})
        context.update(extract_sequence_summary(gaze_values, "gaze"))
    else:
        for column in [
            "gaze_mean",
            "gaze_std",
            "gaze_var",
            "gaze_max",
            "gaze_min",
            "gaze_peak_count",
            "gaze_motion_energy",
            "gaze_count",
            "gaze_active_ratio",
            "gaze_transition_count",
        ]:
            context.setdefault(column, np.nan)

    return context


def process_participant(part_path: Path):
    """Create sliding-window rows for one participant."""
    acc_file = part_path / "E4AccData.csv"
    gsr_file = part_path / "E4GsrData.csv"
    tmp_file = part_path / "E4TmpData.csv"
    if not (acc_file.exists() and gsr_file.exists() and tmp_file.exists()):
        return []

    try:
        acc = pd.read_csv(acc_file)
        gsr = pd.read_csv(gsr_file)
        tmp = pd.read_csv(tmp_file)
        gaze = pd.read_csv(part_path / "GazeData.csv") if (part_path / "GazeData.csv").exists() else None
    except Exception as exc:
        print(f"Failed to read CSVs for {part_path}: {exc}")
        return []

    acc_cols = acc.columns.tolist()
    for name in ["Acc_X", "X", "accX", "x"]:
        if name in acc_cols:
            acc_x = name
            break
    else:
        acc_x = find_numeric_col(acc, exclude=["UnixTime", "SGTime"])

    for name in ["Acc_Y", "Y", "accY", "y"]:
        if name in acc_cols:
            acc_y = name
            break
    else:
        acc_y = find_numeric_col(acc, exclude=["UnixTime", "SGTime", acc_x])

    for name in ["Acc_Z", "Z", "accZ", "z"]:
        if name in acc_cols:
            acc_z = name
            break
    else:
        acc_z = find_numeric_col(acc, exclude=["UnixTime", "SGTime", acc_x, acc_y])

    gsr_col = find_numeric_col(gsr, exclude=["UnixTime", "SGTime"])
    tmp_col = find_numeric_col(tmp, exclude=["UnixTime", "SGTime"])
    gaze_col = find_numeric_col(gaze, exclude=["UnixTime", "SGTime"]) if gaze is not None else None

    acc_1hz = resample_to_1hz(acc, time_col="UnixTime")
    gsr_1hz = resample_to_1hz(gsr, time_col="UnixTime")
    tmp_1hz = resample_to_1hz(tmp, time_col="UnixTime")
    gaze_1hz = resample_to_1hz(gaze, time_col="SGTime") if gaze is not None else pd.DataFrame()

    if acc_1hz.empty:
        return []

    df = pd.DataFrame(index=acc_1hz.index)
    if acc_x and acc_x in acc_1hz.columns:
        df["Acc_X"] = acc_1hz[acc_x]
    if acc_y and acc_y in acc_1hz.columns:
        df["Acc_Y"] = acc_1hz[acc_y]
    if acc_z and acc_z in acc_1hz.columns:
        df["Acc_Z"] = acc_1hz[acc_z]
    if {"Acc_X", "Acc_Y", "Acc_Z"}.issubset(df.columns):
        df["Acc_SVM"] = np.sqrt(df["Acc_X"] ** 2 + df["Acc_Y"] ** 2 + df["Acc_Z"] ** 2)

    if gsr_col and gsr_col in gsr_1hz.columns:
        df = df.join(gsr_1hz[[gsr_col]], how="left")
        df = df.rename(columns={gsr_col: "GSR"})
    if tmp_col and tmp_col in tmp_1hz.columns:
        df = df.join(tmp_1hz[[tmp_col]], how="left")
        df = df.rename(columns={tmp_col: "Temperature"})
    if gaze_col and gaze_col in gaze_1hz.columns:
        df = df.join(gaze_1hz[[gaze_col]], how="left")
        df = df.rename(columns={gaze_col: "Gaze"})

    df = df.dropna(how="all")
    if df.empty:
        return []

    participant_context = build_context_features(
        {"gaze": gaze},
        part_path.parents[1],
        normalize_condition_name(part_path.parent.name),
        part_path.name,
    )

    rows = []
    for start_time, features in window_features(df, window_size=60, step=30):
        row = dict(features)
        row.update(participant_context)
        row["participant"] = part_path.name
        row["start_time"] = start_time
        row["label"] = np.nan
        rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/Engagnition Dataset", help="Root folder containing condition subfolders")
    parser.add_argument("--out", default="data/processed_features.csv")
    args = parser.parse_args()

    root = Path(args.data_root)
    if not root.exists():
        print("Data root not found:", root)
        return

    rows = []
    for condition in root.iterdir():
        if not condition.is_dir():
            continue
        for participant in condition.iterdir():
            if not participant.is_dir():
                continue
            print("Processing", condition.name, participant.name)
            participant_rows = process_participant(participant)
            for row in participant_rows:
                row["condition"] = condition.name
                row["label"] = infer_three_class_label(condition.name)
                row["label_name"] = label_name(row["label"])
            rows.extend(participant_rows)

    if not rows:
        print("No features extracted.")
        return

    df = pd.DataFrame(rows)

    model_cols_path = ROOT / "models" / "feature_columns.joblib"
    if model_cols_path.exists():
        try:
            feature_cols = joblib.load(model_cols_path)
        except Exception:
            feature_cols = [column for column in df.columns if column not in {"participant", "condition", "start_time", "label", "label_name"}]
    else:
        feature_cols = [column for column in df.columns if column not in {"participant", "condition", "start_time", "label", "label_name"}]

    ordered_cols = ["participant", "condition", "start_time", "label", "label_name"] + [column for column in feature_cols if column in df.columns]
    ordered_cols = [column for column in ordered_cols if column in df.columns]
    df = df[ordered_cols]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print("Wrote", out_path, "rows=", len(df))


if __name__ == "__main__":
    main()
