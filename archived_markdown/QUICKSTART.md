# Quick Start Guide

## Get Results in 5 Minutes

### 1. Install Dependencies
```bash
pip install pandas numpy scipy scikit-learn matplotlib joblib jupyter
```

### 2. Launch Notebook
```bash
cd notebooks
jupyter notebook research.ipynb
```

### 3. Run Pipeline
Execute all cells (Kernel → Restart & Run All)
- Takes ~2-3 minutes
- Processes 57 participants
- Trains Random Forest classifier
- Generates 86% accurate model

### 4. Check Results
After execution, find outputs in:
```
models/
├── engagement_recognition_model.joblib     (Trained model)
└── feature_columns.joblib                   (Feature names)

results/
├── feature_importance.csv                   (Feature scores)
├── feature_importance.png                   (Visualization)
└── model_metrics.joblib                     (Eval metrics)
```
