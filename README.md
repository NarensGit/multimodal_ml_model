# Multimodal ML Model — Quick Start

Overview
- This repository contains scripts and models for engagement recognition from wearable sensor data.

Prerequisites
- Python 3.8+ and `pip`
- Create a virtual environment: `python -m venv .venv` then `source .venv/bin/activate` (macOS/Linux)
- Install dependencies: `pip install -r requirements.txt`

Quick Commands
- Prepare processed dataset: `python create_processed_dataset.py`
- Create binary dataset: `python create_binary_dataset.py`
- Train model: `python train_binary_model.py` (outputs saved under `models/`)
- Run the web app / inference: `python app.py` (templates in `templates/`, results in `results/`)
- Run smoke tests: `python smoke_test.py`
- Run full tests: `python test_model.py`

Files & Folders
- `data/` — raw and organized input files
- `models/` — trained model and `feature_columns.joblib`
- `results/` — test outputs and prediction CSVs
- `templates/` — HTML templates for results display
- `uploads/` — sample upload files for quick inference

Notes
- Sample features are in `uploads/sample_features.csv` and `sample_features.csv`.
- Pretrained model: `models/engagement_recognition_model.joblib` can be used for quick inference.
- Logs: `train_log.txt` and `preprocess_log.txt` contain processing/training details.

Next steps
- To update or expand these instructions, ask to refine examples or add platform-specific guides.

Contact
- For questions about usage or data format, open an issue or message the repo owner.
