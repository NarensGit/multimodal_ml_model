# API Documentation

## Data Loading Functions

### `load_participant_data(participant_path)`
Loads physiological data for a single participant.

**Parameters:**
- `participant_path` (Path): Path to participant folder

**Returns:**
- `dict` with keys `'gsr'`, `'acc'`, `'tmp'` containing DataFrames
- `None` if any file is missing

**Example:**
```python
from pathlib import Path
data = load_participant_data(Path("data/Engagnition Dataset/Baseline condition/P01"))
print(data['gsr'].head())  # GSR DataFrame
```

### `collect_all_data()`
Collects data from all participants across all conditions.

**Parameters:** None

**Returns:**
- List of dicts with keys: `participant_id`, `condition`, `label`, `data`

**Example:**
```python
all_data = collect_all_data()
print(f"Found {len(all_data)} participants")
```

---

## Signal Processing Functions

### `synchronize_and_merge_signals(gsr_df, acc_df, tmp_df)`
Aligns and merges multimodal signals with different sampling rates.

**Parameters:**
- `gsr_df` (DataFrame): GSR data with UnixTime, GSR columns
- `acc_df` (DataFrame): Accelerometer data with UnixTime, Acc_X/Y/Z/SVM columns
- `tmp_df` (DataFrame): Temperature data with UnixTime, ST columns

**Returns:**
- DataFrame with columns: UnixTime, GSR, Acc_X, Acc_Y, Acc_Z, Acc_SVM, Temperature

### `handle_missing_values(df, method='forward_fill')`
Handles missing values in time series data.

**Parameters:**
- `df` (DataFrame): Input data
- `method` (str): 'forward_fill', 'interpolate', or 'drop'

**Returns:**
- DataFrame with missing values handled

### `normalize_signals(df, exclude_cols=['UnixTime'])`
Normalizes signals using StandardScaler.

**Returns:**
- Tuple: (scaled_df, scaler_object)

---

## Feature Extraction Functions

### `create_sliding_windows(df, window_size=60, step_size=30)`
Creates overlapping time windows from signal data.

### `extract_statistical_features(window)`
Extracts 7 statistical features from a signal window: mean, std, var, max, min, peak_count, motion_energy.

### `extract_features_from_windows(windows)`
Batch processes multiple windows to extract features and returns a list/dict suitable for DataFrame conversion.

---

## Processing Pipeline Functions

### `process_participant_complete(participant_info)`
End-to-end processing for a single participant. Merges signals, handles missing values, normalizes, windows, extracts features, and labels outputs.

---

## Model Functions

### `predict_engagement(gsr_data, acc_data, tmp_data, model_path, feature_cols_path)`
Makes engagement predictions on new data.

**Returns:**
- Dict with keys: `predictions`, `prediction_codes`, `probabilities`, `n_windows`, `dominant_engagement`, `confidence`

**Example:**
```python
results = predict_engagement(gsr, acc, tmp, 
                            "models/engagement_recognition_model.joblib",
                            "models/feature_columns.joblib")
print(results['dominant_engagement'])
```

---

## Data Format Specifications

Input CSV formats expected: `E4GsrData.csv`, `E4AccData.csv`, `E4TmpData.csv` with required columns `UnixTime` plus sensor columns.

---
