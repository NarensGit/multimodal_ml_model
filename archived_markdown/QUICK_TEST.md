# ⚡ QUICK TEST REFERENCE

## 🔥 Fastest Way to Test (30 Seconds)

```python
# In Jupyter notebook, run these cells in order:

# Cell 1: Load everything (already done if notebook ran)
import joblib
model = joblib.load("../models/engagement_recognition_model.joblib")

# Cell 2: Test on existing data
predictions = model.predict(X_test)
accuracy = (predictions == y_test).mean()
print(f"✓ Accuracy: {accuracy:.1%}")
```

**That's it!** You'll see: `✓ Accuracy: 86.0%`

---

## 🧪 Test with New Participant Data (1 Minute)

```python
# Load one participant
participant_path = "../data/Engagnition Dataset/HPE condition/P39"
gsr = pd.read_csv(f"{participant_path}/E4GsrData.csv")
acc = pd.read_csv(f"{participant_path}/E4AccData.csv")
tmp = pd.read_csv(f"{participant_path}/E4TmpData.csv")

# Predict
results = predict_engagement(gsr, acc, tmp, 
                            "../models/engagement_recognition_model.joblib",
                            "../models/feature_columns.joblib")

# View result
print(results['dominant_engagement'])
print(f"Confidence: {results['confidence']:.1%}")
```
