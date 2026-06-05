# 🧪 MODEL TESTING GUIDE

Complete guide to test the trained engagement recognition model with new data.

---

## 📋 Quick Start (2 Minutes)

### **Method 1: Test with Existing Test Set** (Fastest)
```python
import joblib
from pathlib import Path

# Load model and features
model_path = Path("../models/engagement_recognition_model.joblib")
feature_cols_path = Path("../models/feature_columns.joblib")

model = joblib.load(model_path)
feature_cols = joblib.load(feature_cols_path)

# Use test data already in notebook
predictions = model.predict(X_test)
accuracy = (predictions == y_test).mean()
print(f"✓ Test accuracy: {accuracy:.1%}")
```

### **Method 2: Test with New CSV Data** (Recommended)
```python
import pandas as pd

# Load new participant data
gsr_df = pd.read_csv("path/to/E4GsrData.csv")
acc_df = pd.read_csv("path/to/E4AccData.csv")
tmp_df = pd.read_csv("path/to/E4TmpData.csv")

# Use built-in prediction function
results = predict_engagement(gsr_df, acc_df, tmp_df, model_path, feature_cols_path)
print(results['dominant_engagement'])
print(f"Confidence: {results['confidence']:.1%}")
```

---

## 🎯 Detailed Testing Methods

### **Method A: Test with Single Sample from Test Set**

```python
# Get one sample
sample_idx = 0
X_sample = X_test[sample_idx].reshape(1, -1)
y_true = y_test[sample_idx]

# Make prediction
prediction = model.predict(X_sample)[0]
probability = model.predict_proba(X_sample)[0]

# Display results
label_map = {0: "Baseline", 1: "Low Engagement", 2: "High Engagement"}
print(f"True Label:     {label_map[y_true]}")
print(f"Predicted:      {label_map[prediction]}")
print(f"Match: {'✓' if y_true == prediction else '✗'}")
print(f"\nProbabilities:")
for i, prob in enumerate(probability):
    print(f"  {label_map[i]}: {prob:.1%}")
```

---

### **Method B: Test All Test Samples with Summary**

```python
# Make predictions on all test samples
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Calculate metrics
accuracy = (predictions == y_test).mean()
correct = (predictions == y_test).sum()
total = len(y_test)

print(f"✓ Test Results:")
print(f"  Accuracy: {accuracy:.1%} ({correct}/{total})")
print(f"  Total samples: {total}")
print(f"\nCorrect predictions: {correct}")
print(f"Wrong predictions: {total - correct}")
```

---

### **Method C: Test with New Participant Data (Real CSV Files)**

#### **Step 1: Load Raw CSV Files**

```python
import pandas as pd
from pathlib import Path

# Path to a participant folder
participant_path = Path("../data/Engagnition Dataset/HPE condition/P39")

# Load raw sensor data
gsr_df = pd.read_csv(participant_path / "E4GsrData.csv")
acc_df = pd.read_csv(participant_path / "E4AccData.csv")
tmp_df = pd.read_csv(participant_path / "E4TmpData.csv")

print(f"✓ GSR shape: {gsr_df.shape}")
print(f"✓ Accelerometer shape: {acc_df.shape}")
print(f"✓ Temperature shape: {tmp_df.shape}")
```

#### **Step 2: Run Prediction**

```python
# Make engagement prediction
results = predict_engagement(gsr_df, acc_df, tmp_df, model_path, feature_cols_path)

# Display results
print(f"✓ Predictions complete!")
print(f"\nDominant Engagement: {results['dominant_engagement']}")
print(f"Confidence: {results['confidence']:.1%}")
print(f"Total windows analyzed: {results['n_windows']}")
print(f"\nPer-window predictions:")
for i, pred in enumerate(results['predictions'][:5]):  # Show first 5
    print(f"  Window {i+1}: {pred}")
```

---

### **Method D: Test Multiple Participants in Batch**

```python
import pandas as pd
from pathlib import Path

results_summary = []

# Test all HPE participants
hpe_condition = Path("../data/Engagnition Dataset/HPE condition")

for participant_folder in sorted(hpe_condition.iterdir())[:5]:  # First 5
    if not participant_folder.is_dir():
        continue
    
    # Load data
    gsr_df = pd.read_csv(participant_folder / "E4GsrData.csv")
    acc_df = pd.read_csv(participant_folder / "E4AccData.csv")
    tmp_df = pd.read_csv(participant_folder / "E4TmpData.csv")
    
    # Predict
    results = predict_engagement(gsr_df, acc_df, tmp_df, model_path, feature_cols_path)
    
    # Store results
    results_summary.append({
        'participant': participant_folder.name,
        'engagement': results['dominant_engagement'],
        'confidence': results['confidence'],
        'n_windows': results['n_windows']
    })

# Display summary
df_summary = pd.DataFrame(results_summary)
print(df_summary.to_string(index=False))
```

---

### **Method E: Visualize Predictions**

```python
import matplotlib.pyplot as plt

# Get predictions with probabilities
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: Prediction distribution
from collections import Counter
pred_counts = Counter(predictions)
labels = ["Baseline", "Low Engagement", "High Engagement"]
counts = [pred_counts.get(i, 0) for i in range(3)]

axes[0].bar(labels, counts, color=['green', 'orange', 'red'])
axes[0].set_ylabel("Number of Predictions")
axes[0].set_title("Model Predictions Distribution")
axes[0].grid(axis='y', alpha=0.3)

# Plot 2: Confidence distribution
confidences = [max(prob) for prob in probabilities]
axes[1].hist(confidences, bins=15, edgecolor='black', alpha=0.7)
axes[1].set_xlabel("Prediction Confidence")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Model Confidence Distribution")
axes[1].axvline(np.mean(confidences), color='red', linestyle='--', label=f"Mean: {np.mean(confidences):.2f}")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("../results/prediction_analysis.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"✓ Confidence statistics:")
print(f"  Mean: {np.mean(confidences):.1%}")
print(f"  Min: {np.min(confidences):.1%}")
print(f"  Max: {np.max(confidences):.1%}")
```

---

### **Method F: Test with Sample Synthetic Data**

```python
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Create synthetic sensor data (2 minutes at different rates)
# Simulate 120 seconds of data

# GSR at 4 Hz (1 sample per 0.25s)
gsr_time = np.arange(0, 120, 0.25)
gsr_values = np.random.normal(2.5, 0.8, len(gsr_time))  # Mean=2.5, std=0.8
gsr_df = pd.DataFrame({
    'UnixTime': 1700000000 + gsr_time,
    'GSR': np.clip(gsr_values, 0.1, 5)  # Realistic range
})

# Accelerometer at 32 Hz (1 sample per 0.03125s)
acc_time = np.arange(0, 120, 1/32)
acc_x = np.random.normal(0.1, 0.5, len(acc_time))
acc_y = np.random.normal(0.2, 0.6, len(acc_time))
acc_z = np.random.normal(9.7, 0.3, len(acc_time))
acc_svm = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

acc_df = pd.DataFrame({
    'UnixTime': 1700000000 + acc_time,
    'Acc_X': acc_x,
    'Acc_Y': acc_y,
    'Acc_Z': acc_z,
    'Acc_SVM': acc_svm
})

# Temperature at 4 Hz
tmp_time = np.arange(0, 120, 0.25)
tmp_values = np.random.normal(36.5, 0.3, len(tmp_time))
tmp_df = pd.DataFrame({
    'UnixTime': 1700000000 + tmp_time,
    'ST': np.clip(tmp_values, 35, 38)  # Realistic skin temp
})

# Make prediction
results = predict_engagement(gsr_df, acc_df, tmp_df, model_path, feature_cols_path)
print(f"✓ Synthetic data prediction:")
print(f"  Engagement: {results['dominant_engagement']}")
print(f"  Confidence: {results['confidence']:.1%}")
```
