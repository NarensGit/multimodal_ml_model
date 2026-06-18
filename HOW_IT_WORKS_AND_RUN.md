# How This Project Works and How to Run It

## What the current model does
This project trains a 3-class engagement classifier for the Engagnition dataset:
- `0` = Baseline
- `1` = Low Engagement (LPE)
- `2` = High Engagement (HPE)

The pipeline uses the raw sensor files in each participant folder and extracts sliding-window features from:
- GSR
- Accelerometer
- Temperature
- Optional gaze data
- Optional session elapsed time metadata
- Optional intervention metadata

## How it works
1. `create_processed_dataset.py` reads the raw dataset folders under `data/Engagnition Dataset`.
2. It resamples the signals to 1 Hz and creates sliding windows.
3. It extracts summary features from each window.
4. It adds optional gaze, elapsed-time, and intervention features when those files are present.
5. It assigns the 3-class label from the condition folder name.
6. `train_three_class_model.py` trains a RandomForest classifier on the extracted features.
7. The model and feature column order are saved to `models/` so inference can reuse them.
8. `app.py` and `test_model.py` load the saved model and make predictions on new CSV files.

## How to run
### 1. Install dependencies
```bash
pip install -r requirements.txt

```

### 2. Rebuild processed features from raw data
```bash
python create_processed_dataset.py --data-root "data/Engagnition Dataset" --out data/processed_features.csv
```

### 3. Train the 3-class model
```bash
python train_three_class_model.py --data "data/Engagnition Dataset" --feature-set all
```

You can also compare feature combinations:
```bash
python train_three_class_model.py --data "data/Engagnition Dataset" --feature-set sensor_only
python train_three_class_model.py --data "data/Engagnition Dataset" --feature-set sensor_plus_gaze
python train_three_class_model.py --data "data/Engagnition Dataset" --feature-set sensor_plus_context
```

### 4. Run a quick sanity check
```bash
python smoke_test.py
```

### 5. Predict from a CSV
```bash
python test_model.py --features sample_features.csv --out results/predictions.csv
```

### 6. Run the web app
```bash
python app.py
```
Then open the local address shown in the terminal and upload a feature CSV.

## Expected outputs
After training, these files are updated:
- `models/engagement_recognition_model.joblib`
- `models/feature_columns.joblib`
- `results/model_metrics.joblib`
- `results/feature_importance_3class.csv`

## Notes
- The model expects the same feature columns saved during training.
- If a CSV is missing some features, the app and CLI now pad missing columns with zeros.
- The new pipeline is 3-class, not binary.
