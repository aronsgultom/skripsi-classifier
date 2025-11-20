# ============================================================
# 1) APLIKASI FLASK UNTUK KLASIFIKASI PDF (VERSI RAILWAY)
# ============================================================

import os
import numpy as np
import joblib
import json
import warnings
from flask import Flask, request, jsonify, render_template

warnings.filterwarnings("ignore")

# ============================================================
# 2) IMPORT FUNGSI MODEL
# ============================================================

from config import Config

try:
    from train_model import (
        ekstrak_teks_pdf,
        pra_proses_teks,
        tokens_namafile,
        ClassCentroidCosineClassifier
    )
except ImportError as e:
    print(f"[FATAL ERROR] train_model.py gagal di-import: {e}")

    # Dummy fallback supaya server tidak crash
    class ClassCentroidCosineClassifier:
        def __init__(self): pass
    def ekstrak_teks_pdf(x): return ""
    def pra_proses_teks(x): return x
    def tokens_namafile(x): return ""

# ============================================================
# 3) FLASK SETUP
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, root_path=BASE_DIR)

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

# ============================================================
# 4) MODEL PATH
# ============================================================

MODEL_DIR = getattr(Config, "MODEL_DIR", os.path.join(BASE_DIR, "models"))
MODEL_PATH = os.path.join(MODEL_DIR, "final_model.pkl")
CLASSES_PATH = os.path.join(MODEL_DIR, "classes.json")

best_final_model = None
classes = None

# ============================================================
# 5) LOAD MODEL
# ============================================================

def load_saved_model():
    global best_final_model, classes

    if not os.path.exists(MODEL_PATH):
        print("[WARNING] Model tidak ditemukan:", MODEL_PATH)
        return

    if not os.path.exists(CLASSES_PATH):
        print("[WARNING] classes.json tidak ditemukan:", CLASSES_PATH)
        return

    try:
        best_final_model = joblib.load(MODEL_PATH)

        with open(CLASSES_PATH, "r", encoding="utf-8") as f:
            classes = json.load(f)

        print("[INFO] Model berhasil dimuat:", MODEL_PATH)
        print("[INFO] Jumlah kelas:", len(classes))

    except Exception as e:
        print("[FATAL] Gagal memuat model:", e)
        best_final_model = None
        classes = None

# Load saat server boot
load_saved_model()

# ============================================================
# 6) ROUTES
# ============================================================

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

    # Validasi file
    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file dalam request"}), 400

    file = request.files["file"]

    if file.filename == "" or not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Harap unggah file PDF"}), 400

    filename = os.path.basename(file.filename)
    saved_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    try:
        file.save(saved_path)
    except Exception as e:
        return jsonify({"error": f"Gagal menyimpan file: {e}"}), 500

    # Proses klasifikasi
    try:
        teks = ekstrak_teks_pdf(saved_path)

        if not teks.strip():
            return jsonify({
                "error": "Teks kosong. PDF mungkin adalah gambar atau OCR tidak tersedia."
            }), 400

        teks_final = teks + " " + tokens_namafile(saved_path)
        teks_bersih = pra_proses_teks(teks_final)

        proba = best_final_model.predict_proba([teks_bersih])[0]
        pred_idx = int(np.argmax(proba))
        pred_label = classes[pred_idx]

        order = np.argsort(proba)[::-1]
        top5 = [
            {"label": classes[i], "prob": float(proba[i])}
            for i in order[:5]
        ]

        return jsonify({
            "file": filename,
            "prediction": pred_label,
            "score": float(proba[pred_idx]),
            "top_results": top5
        })

    except Exception as e:
        print("[ERROR]", e)
        return jsonify({"error": "Kesalahan internal server"}), 500

    finally:
        try:
            if os.path.exists(saved_path):
                os.remove(saved_path)
        except:
            pass

# ============================================================
# 7) ENTRY POINT
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Railway requires this
    app.run(host="0.0.0.0", port=port)
