from flask import Flask, request, render_template, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import os

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "engagement_recognition_model.joblib"
COLS_PATH = ROOT / "models" / "feature_columns.joblib"
UPLOAD_FOLDER = ROOT / "uploads"
RESULTS_FOLDER = ROOT / "results"
ALLOWED_EXTENSIONS = {"csv"}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULTS_FOLDER.mkdir(exist_ok=True)

model = None
cols = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load model at import time (ensures compatibility across Flask versions)
try:
    if not MODEL_PATH.exists() or not COLS_PATH.exists():
        raise RuntimeError(f"Model or columns not found in models/ folder: {MODEL_PATH}, {COLS_PATH}")
    model = joblib.load(MODEL_PATH)
    cols = joblib.load(COLS_PATH)
except Exception as e:
    # Defer the error until first request if model can't be loaded now
    model = None
    cols = None
    load_error = e
else:
    load_error = None


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"success": False, "error": f"Model not loaded: {load_error}"}), 500

    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = Path(app.config['UPLOAD_FOLDER']) / filename
        file.save(filepath)
        
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            return jsonify({"success": False, "error": f"Failed to parse CSV: {str(e)}"}), 400
            
        if df.empty:
            return jsonify({"success": False, "error": "Uploaded CSV file is empty"}), 400
            
        X = df.reindex(columns=cols, fill_value=0).fillna(0)
        preds = model.predict(X)
        results_df = df.copy()
        
        # numeric label
        results_df['predicted_label'] = preds
        # readable textual label
        LABEL_MAP = {0: 'baseline', 1: 'low', 2: 'high'}
        results_df['predicted_label_name'] = results_df['predicted_label'].map(LABEL_MAP).fillna(results_df['predicted_label'].astype(str))
        
        probs = None
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X)
            for i, cls in enumerate(model.classes_):
                results_df[f'prob_{cls}'] = probs[:, i]
                
        out_path = RESULTS_FOLDER / f"predictions_{filename}"
        results_df.to_csv(out_path, index=False)
        
        # Calculate overall engagement level and percentage distributions
        preds_series = pd.Series(preds)
        total_preds = len(preds)
        
        # Calculate percentage distributions
        pct_baseline = float((preds_series == 0).sum() / total_preds * 100)
        pct_low = float((preds_series == 1).sum() / total_preds * 100)
        pct_high = float((preds_series == 2).sum() / total_preds * 100)
        
        # The overall engagement level is determined by the majority class
        mode_val = preds_series.mode()
        overall_label_num = int(mode_val.iloc[0]) if not mode_val.empty else 0
        overall_engagement = LABEL_MAP.get(overall_label_num, str(overall_label_num))
        
        # Build list of rows with details
        rows_data = []
        for index, row in results_df.iterrows():
            row_dict = {
                "index": int(index + 1),
                "start_time": float(row["start_time"]) if "start_time" in row else float(index * 30),
                "participant": str(row["participant"]) if "participant" in row else "Unknown",
                "prediction": LABEL_MAP.get(int(row["predicted_label"]), str(row["predicted_label"])),
            }
            if probs is not None:
                row_dict["probabilities"] = {
                    LABEL_MAP.get(cls, str(cls)): float(row[f'prob_{cls}']) for cls in model.classes_
                }
            rows_data.append(row_dict)
            
        return jsonify({
            "success": True,
            "overall_engagement": overall_engagement,
            "distribution": {
                "baseline": pct_baseline,
                "low": pct_low,
                "high": pct_high
            },
            "download_url": f"/results/{out_path.name}",
            "rows": rows_data
        })
        
    return jsonify({"success": False, "error": "File type not allowed"}), 400



@app.route('/results/<path:filename>')
def download_result(filename):
    return send_from_directory(str(RESULTS_FOLDER), filename, as_attachment=True)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(host='127.0.0.1', port=port)

