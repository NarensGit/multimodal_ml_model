**Prediction Explanation**

This note explains how the saved model predicts engagement, why the current outputs are all appearing as `disengaged`, and actionable next steps to address this.

**What the predictor does**
- Input: a CSV with feature columns in the same order as `models/feature_columns.joblib` (example: [data/test_upload_features.csv](data/test_upload_features.csv)).
- The model loaded from `models/engagement_recognition_model.joblib` is a scikit-learn classifier with `classes_ = [0, 1]` where we map 0 -> `disengaged` and 1 -> `engaged` in both CLI and the Flask UI.
- The code fills missing values with zero, calls `model.predict(X)` to get numeric labels, and (if available) `model.predict_proba(X)` to get class probabilities. The UI saves `predicted_label` (numeric) and `predicted_label_name` (text).

**What I observed (empirical evidence)**
- Model classes: [0, 1]
- I loaded `data/test_upload_features.csv` and computed prediction probabilities for each row.
- Summary of probabilities across the 25 synthetic rows:
  - mean probability for class 0: 0.8854747705973618 (std 0.0011211032620055427)
  - mean probability for class 1: 0.1145252294026385 (std 0.001121103262005556)
- Example first 10 predicted probability vectors (class 0, class 1):
  - [0.88461749 0.11538251]
  - [0.88669931 0.11330069]
  - [0.88461749 0.11538251]
  - [0.88669931 0.11330069]
  - [0.88461749 0.11538251]
  - [0.88669931 0.11330069]
  - [0.88461749 0.11538251]
  - [0.88767989 0.11232011]
  - [0.88461749 0.11538251]
  - [0.88461749 0.11538251]

Because the model predicts class 0 with ~88% probability for every row, the final label (the argmax of probabilities or `model.predict`) is 0 for all rows, which we display as `disengaged`.

**Why this happens (possible causes)**
1. Class preference / imbalance in training data
   - The model may have learned that class 0 is more likely; even if training was balanced by undersampling, the learned decision boundary can still favour class 0 for feature vectors similar to many inputs.
2. Synthetic input distribution
   - The synthetic CSV (`data/test_upload_features.csv`) may not contain feature patterns the model recognizes as `engaged` — many rows have relatively low activity/motion features, so the model assigns low probability to class 1.
3. Feature scaling / order mismatch
   - If feature order is wrong or scaling differs from training (e.g., model expected normalized features but input is raw), predictions will be impacted. We validated that the header matches the saved `feature_columns.joblib` when generating the CSV.
4. Decision threshold and probability calibration
   - The model uses the default argmax decision rule. If probabilities are not well calibrated, small differences won't flip the predicted class. Calibration (Platt scaling/Isotonic) can help.

**Quick checks you can run now**
- Inspect saved probabilities in a predictions CSV: `results/test_predictions.csv` (or `results/predictions_<upload>.csv`) — columns `prob_0` and `prob_1` are present.
- Confirm the feature column order: `models/feature_columns.joblib` vs your CSV header (the CLI will print missing columns if they don't match).

**Concrete next steps to reduce all-`disengaged` outputs**
1. Sanity-check features
   - Verify `data/test_upload_features.csv` header matches exactly the saved feature order: open `models/feature_columns.joblib` and compare.
2. Try a few known positive examples
   - If you have example windows that should be `engaged`, run them and inspect `prob_1`. If those also show low `prob_1`, the model may be underpredicting engaged.
3. Retrain with stronger balancing or weighting
   - Use `class_weight='balanced'` (already used) or oversample `engaged` with SMOTE so the model sees more diverse positive examples.
4. Calibrate probabilities
   - Wrap the classifier with `CalibratedClassifierCV` to obtain better calibrated probabilities, which helps threshold-based decisions.
5. Threshold tuning
   - Instead of argmax, choose a lower threshold for `prob_1` (e.g., predict `engaged` if `prob_1 > 0.3`) if recall for `engaged` needs to improve. Evaluate on validation set to pick threshold.
6. Feature engineering
   - Add or normalize features that discriminate engagement (e.g., higher motion energy, GSR peaks). Try tree-based models like LightGBM or XGBoost.

**How to implement a quick probability-threshold hack**
1. Modify `test_model.py` (or `app.py`) after `probs = model.predict_proba(X)`:

```python
# choose threshold for class 1 (engaged)
threshold = 0.3
preds = (probs[:, 1] >= threshold).astype(int)
```

2. Map `preds` to textual labels as before. This increases `engaged` predictions when `prob_1` is reasonably high but not the argmax.

**Recommendation (safe, effective)**
- First: verify feature order and test a few ground-truth positive windows.
- Second: calibrate probabilities and select a threshold using a validation set. This avoids blind thresholding and keeps performance measurable.
- Third: if ground-truth is limited, retrain with better positive samples or try oversampling.

If you want, I will:
- (A) add a small script that computes and prints `prob_0`/`prob_1` statistics for any uploaded CSV, and exposes a threshold toggle in the Flask UI, or
- (B) retrain and calibrate the classifier and return a new model, or
- (C) implement calibrated probabilities + a configurable threshold in both `test_model.py` and `app.py` and re-run predictions.

Tell me which option you prefer and I will implement it. If you choose retrain, tell me whether to use the existing balanced dataset or try oversampling (SMOTE).

Files referenced:
- Model: [models/engagement_recognition_model.joblib](models/engagement_recognition_model.joblib)
- Feature columns: [models/feature_columns.joblib](models/feature_columns.joblib)
- Synthetic test data: [data/test_upload_features.csv](data/test_upload_features.csv)
- Example predictions: [results/test_predictions.csv](results/test_predictions.csv)
