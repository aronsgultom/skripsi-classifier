=======================
app.py (siap deploy)
=======================

import os
import numpy as np
import joblib
import json
import logging
from flask import Flask, request, jsonify, render_template

=======================
Logging
=======================

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

=======================
Flask setup
=======================

app = Flask(name)

BASE_DIR = os.path.dirname(os.path.abspath(file))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024 # 16 MB

=======================
Model paths
=======================

MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "final_model.pkl")
CLASSES_PATH = os.path.join(MODEL_DIR, "classes.json")

best_final_model = None
classes = None

=======================
Load model
=======================

def load_saved_model():
global best_final_model, classes
if not os.path.exists(MODEL_PATH):
logging.warning(f"Model tidak ditemukan: {MODEL_PATH}")
return
if not os.path.exists(CLASSES_PATH):
logging.warning(f"classes.json tidak ditemukan: {CLASSES_PATH}")
return
try:
best_final_model = joblib.load(MODEL_PATH)
with open(CLASSES_PATH, "r", encoding="utf-8") as f:
classes = json.load(f)
logging.info(f"Model berhasil dimuat: {MODEL_PATH}, jumlah kelas: {len(classes)}")
except Exception as e:
logging.error(f"Gagal memuat model: {e}")
best_final_model = None
classes = None

load_saved_model()

=======================
Import model functions
=======================

try:
from train_model import (
ekstrak_teks_pdf,
pra_proses_teks,
tokens_namafile,
ClassCentroidCosineClassifier
)
except ImportError as e:
logging.error(f"train_model.py gagal di-import: {e}")
# Dummy fallback
class ClassCentroidCosineClassifier:
def init(self): pass
def ekstrak_teks_pdf(x): return ""
def pra_proses_teks(x): return x
def tokens_namafile(x): return ""

=======================
Routes
=======================

@app.route("/")
def index():
if best_final_model is None:
return render_template(
"index.html",
model_error="Model belum dimuat. Pastikan final_model.pkl tersedia."
), 503
return render_template("index.html")

@app.route("/classify", methods=["POST"])
def classify_document():
if best_final_model is None:
return jsonify({"error": "Model tidak tersedia"}), 503
if "file" not in request.files:
return jsonify({"error": "Tidak ada file dalam request"}), 400
file = request.files["file"]
if file.filename == "" or not file.filename.lower().endswith(".pdf"):
return jsonify({"error": "Harap unggah file PDF"}), 400

filename = os.path.basename(file.filename)
saved_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

try:
    file.save(saved_path)
    teks = ekstrak_teks_pdf(saved_path)
    if not teks.strip():
        return jsonify({"error": "Teks kosong. PDF mungkin gambar atau OCR tidak tersedia."}), 400

    teks_final = teks + " " + tokens_namafile(saved_path)
    teks_bersih = pra_proses_teks(teks_final)
    proba = best_final_model.predict_proba([teks_bersih])[0]
    pred_idx = int(np.argmax(proba))
    pred_label = classes[pred_idx]

    order = np.argsort(proba)[::-1]
    top5 = [{"label": classes[i], "prob": float(proba[i])} for i in order[:5]]

    return jsonify({
        "file": filename,
        "prediction": pred_label,
        "score": float(proba[pred_idx]),
        "top_results": top5
    })

except Exception as e:
    logging.error(f"Kesalahan saat klasifikasi: {e}")
    return jsonify({"error": "Kesalahan internal server"}), 500

finally:
    if os.path.exists(saved_path):
        try:
            os.remove(saved_path)
        except Exception:
            pass
