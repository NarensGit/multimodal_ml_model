# Quick Reference Card

## 🚀 Start Here (30 seconds)

```bash
# Install
pip install pandas numpy scipy scikit-learn matplotlib joblib jupyter

# Launch
cd notebooks && jupyter notebook research.ipynb

# Run everything
Kernel → Restart & Run All (takes ~2-3 min)
```

---

## Model At a Glance

- Purpose: Classify engagement level (Baseline/LPE/HPE)
- Input: 60-second physiological signals (GSR, Acc, Temp)
- Output: Class: Baseline (0), Low Engagement (1), High Engagement (2)
- Accuracy: 86% on test set
- Model: Random Forest (100 trees)
- Features: 42 statistical features per window

---

## Quick Use

```python
import joblib
model = joblib.load("models/engagement_recognition_model.joblib")
results = predict_engagement(gsr_data, acc_data, tmp_data,
    "models/engagement_recognition_model.joblib",
    "models/feature_columns.joblib")
print(results['dominant_engagement'], results['confidence'])
```

---

## Top Features

1. Acc_SVM_min (8.9%)
2. Temperature_motion_energy (6.6%)
3. GSR_std (5.6%)

