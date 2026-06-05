from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from pathlib import Path
import joblib
import pandas as pd
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
        return f"Model not loaded: {load_error}", 500

    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = Path(app.config['UPLOAD_FOLDER']) / filename
        file.save(filepath)
        df = pd.read_csv(filepath)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            return render_template('results.html', error=True, message=f"Missing columns: {missing}")
        X = df[cols].fillna(0)
        preds = model.predict(X)
        results_df = df.copy()
        # numeric label
        results_df['predicted_label'] = preds
        # readable textual label in lowercase as requested
        LABEL_MAP = {0: 'disengaged', 1: 'engaged'}
        results_df['predicted_label_name'] = results_df['predicted_label'].map(LABEL_MAP).fillna(results_df['predicted_label'].astype(str))
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X)
            for i, cls in enumerate(model.classes_):
                results_df[f'prob_{cls}'] = probs[:, i]
        out_path = RESULTS_FOLDER / f"predictions_{filename}"
        results_df.to_csv(out_path, index=False)
        return render_template('results.html', error=False, table=results_df.head(50).to_html(classes='table table-striped', index=False, escape=False), download=str(out_path.name))
    return "File type not allowed", 400


@app.route('/results/<path:filename>')
def download_result(filename):
    return send_from_directory(str(RESULTS_FOLDER), filename, as_attachment=True)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
