# Technical Implementation (simple)

This document summarizes the pipeline I implemented and gives clear, copy-paste steps an ML engineer can use to reproduce everything.

## Goal
- Build a binary engagement classifier (Engaged / Not Engaged) from Empatica E4 sensor data (GSR, accelerometer, temperature).

## What’s in the repo (short)
- Raw data: `data/Engagnition Dataset/<condition>/<participant>/` (contains E4 CSV files).
- Small test file: `sample_features.csv` (already in feature format).
- Model artifacts: `models/engagement_recognition_model.joblib`, `models/feature_columns.joblib`.
- Scripts: preprocessing, dataset creation, training, prediction, and a small Flask UI.

## Pipeline (high level)
1. Read raw E4 CSVs per participant.
2. Convert times to datetime, resample/interpolate signals to 1 Hz.
3. Create sliding windows (60s window, 30s step).
4. Extract 42 features per window (mean, std, var, max, min, peak_count, motion_energy for 6 signals).
5. Map experimental condition to binary label: `HPE condition` -> Engaged (1); `Baseline`/`LPE` -> Not Engaged (0).
6. Balance classes by undersampling the majority class.
7. Train a Random Forest binary classifier and save model + feature order + metrics.

## Key scripts (what they do)
- `create_processed_dataset.py` — raw E4 CSVs -> per-window features (resampling, windowing, feature extraction).
- `create_binary_dataset.py` — condition labels -> binary target, class balancing, saves CSV.
- `train_binary_model.py` — trains RandomForest on the binary dataset or the raw folder (it will build features on the fly), saves model and metrics to `models/` and `results/`.
- `test_model.py` — CLI: load saved model + feature columns, predict on a features CSV, save predictions to CSV (includes readable label names).
- `app.py` + `templates/` — simple Flask app to upload a feature CSV and download predictions.
- `smoke_test.py` — quick script to confirm model loads and predicts.

## Reproduce: exact commands
1. Create virtual environment and install deps:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. (Optional) Recreate features from raw data:
```bash
python create_processed_dataset.py --data-root "data/Engagnition Dataset" --out data/processed_features.csv
```

3. Create balanced binary dataset (if you don't want to train from raw folder):
```bash
python create_binary_dataset.py --in uploads/processed_features.csv --out data/engagement_binary_dataset.csv
```

4. Train the binary model (this script accepts the raw folder or a CSV):
```bash
python train_binary_model.py --data "data/Engagnition Dataset"
```

5. Run a quick prediction (CLI):
```bash
python test_model.py --features sample_features.csv --out results/predictions.csv
```

6. Or use the web UI:
```bash
python app.py
# open http://127.0.0.1:5000 and upload a features CSV
```

## Model & metrics (example)
- Model: RandomForestClassifier (n_estimators=200, max_depth=15, min_samples_split=10, min_samples_leaf=5, class_weight='balanced')
- Example test accuracy after retraining: 0.9032
- Example confusion matrix (test):
```
[[41  3]
 [ 6 43]]
```

## Important notes for the ML engineer
- Labels are proxy labels derived from experimental condition. If human-labeled engagement timestamps are available, align those labels to the window start times and retrain.
- Balancing was done via undersampling. Alternatives: SMOTE, weighted loss, or collecting more Not Engaged samples.
- The model and feature columns are saved with `joblib`. If you change scikit-learn versions, prefer retraining to avoid pickle incompatibilities.

## Suggested next steps (optional)
- Add human annotation alignment when labels exist.
- Try LightGBM/XGBoost and hyperparameter search.
- Containerize (Dockerfile) and provide a one-command run for training + serving.

---

If you want, I will add a Dockerfile and a short `README.md` with one-line run commands next. Tell me if you prefer Docker or a simple bash script for reproducibility.
