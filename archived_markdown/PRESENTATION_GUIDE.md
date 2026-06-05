# 🎤 ELEVATOR PITCH & PRESENTATION GUIDE

## 30-Second Elevator Pitch

"I developed an automated machine learning system that recognizes engagement levels in children with autism using physiological signals. The pipeline processes data from 57 participants across three engagement conditions—capturing GSR, accelerometer, and temperature readings. I engineered a Random Forest classifier that achieves 86% accuracy, with perfect detection of high engagement moments. The model is fully trained, documented, and ready for real-time deployment."

---

## 2-Minute Version

"My project addresses an important challenge in autism research: real-time engagement recognition.

The Problem: Researchers need an automated way to classify engagement levels during therapy sessions using physiological signals.

My Solution: I built a complete ML pipeline that:
- Automatically discovers and processes data from 57 participants
- Merges three sensor modalities with different sampling rates
- Extracts 42 meaningful statistical features
- Trains a Random Forest classifier achieving 86% accuracy

What Makes It Special:
- 100% detection rate for high engagement (never misses an engaged moment)
- Completely automated—no manual intervention
- Production-ready with saved model and full documentation
- Well-engineered with signal synchronization and robust preprocessing
"

---

## 5-Minute Deep Dive (Structure)

1. Problem & Motivation
2. Data & Approach
3. Technical Implementation (synchronization, windows, features)
4. Results & Performance (86.02% accuracy, confusion matrix)
5. Key Insights and Next Steps

---

## Talking Points
- Strengths: automation, synchronization, solid performance
- Why Random Forest: interpretable, fast, good for small datasets
- Improvements: SMOTE, spectral features, LSTM for sequences

---

## Visual Aids
- `feature_importance.png` — show top features
- `research.ipynb` — demo pipeline
- `QUICK_REFERENCE.md` — one-page summary

