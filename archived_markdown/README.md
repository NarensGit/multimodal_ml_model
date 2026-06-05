# Multimodal ML Model for Engagement Recognition in Children with Autism Spectrum Disorder

## Project Overview

This project implements a complete end-to-end machine learning pipeline for **real-time engagement recognition** using multimodal physiological and behavioral signals from children with Autism Spectrum Disorder (ASD).

### Key Features
- **Multimodal Data Integration**: Combines Galvanic Skin Response (GSR), accelerometer, and temperature signals
- **Automatic Data Pipeline**: Dynamically loads and processes all participant data
- **Statistical Feature Extraction**: Generates 42 features from 60-second sliding windows
- **Robust ML Model**: Random Forest classifier achieving 86% test accuracy
- **Production-Ready**: Trained model saved and ready for inference on new data

## Dataset Structure

The project uses the **Engagnition Dataset** with 57 participants across 3 conditions:

```
data/Engagnition Dataset/
├── Baseline condition/          (Label: 0) - 19 participants
├── LPE condition/               (Label: 1) - 19 participants  
└── HPE condition/               (Label: 2) - 19 participants

Each participant folder contains:
├── E4GsrData.csv              (Galvanic Skin Response)
├── E4AccData.csv              (Accelerometer X, Y, Z, SVM)
└── E4TmpData.csv              (Skin Temperature)
```

## Model Performance

### Overall Metrics
- **Test Accuracy**: 86.02%
- **Test Precision**: 85.48%
- **Test Recall**: 86.02%
- **Test F1-Score**: 85.38%

### Per-Class Performance
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Baseline (0) | 0.8421 | 0.7619 | 0.8000 | 21 |
| Low Engagement (1) | 0.7895 | 0.6522 | 0.7143 | 23 |
| High Engagement (2) | 0.8909 | 1.0000 | 0.9423 | 49 |

### Top Features
The model relies most heavily on motion-based features:
1. **Accelerometer SVM (min)** - 8.88%
2. **Temperature motion energy** - 6.62%
3. **GSR standard deviation** - 5.62%
4. **Accelerometer motion energy** - 4.67%
