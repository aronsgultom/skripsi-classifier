# ============================================================
# app/ml/classifier.py — sinkron dengan train_model.py
# ============================================================

import sys, os, re, json, joblib, numpy as np
from config import Config
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import cosine_similarity

# Pastikan joblib bisa menemukan class custom saat unpickle
sys.modules['__main__'] = sys.modules[__name__]

# ------------------ Class custom (harus sama dg training) ------------------
class ClassCentroidCosineClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, use_softmax=True):
        self.use_softmax = use_softmax

    @staticmethod
    def _toarray(X):
        return X.toarray() if hasattr(X, "toarray") else X

    def fit(self, X, y):
        X = self._toarray(X)
        y = np.array(y)
        self.classes_ = np.unique(y)
        cents = []
        for c in self.classes_:
            Xi = X[y == c]
            mu = Xi.mean(axis=0, keepdims=True)
            denom = np.linalg.norm(mu)
            if denom > 0:
                mu = mu / denom
            cents.append(mu)
        self.centroids_ = np.vstack(cents)
        return self

    def decision_function(self, X):
        X = self._toarray(X)
        return cosine_similarity(X, self.centroids_)

    def predict(self, X):
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        s = self.decision_function(X)
        exp_s = np.exp(s - s.max(axis=1, keepdims=True))
        return exp_s / exp_s.sum(axis=1, keepdims=True)

# ------------------ Pra-proses (samakan dg training) ------------------
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, wordpunct_tokenize
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    _STEMMER = StemmerFactory().create_stemmer()
    _STOP_ID = set(stopwords.words('indonesian'))
except Exception:
    _STEMMER = None
    _STOP_ID = set()

def pra_proses_teks(teks: str) -> str:
    teks = (teks or "").lower()
    teks = re.sub(r'[^\w\s]', ' ', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    if _STEMMER is None or not _STOP_ID:
        return teks
    try:
        tokens = word_tokenize(teks)
    except Exception:
        tokens = wordpunct_tokenize(teks)
    tokens = [w for w in tokens if w not in _STOP_ID and len(w) > 2]
    return _STEMMER.stem(" ".join(tokens))

def tokens_namafile(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r'[_\-\.\$\$\$\$,]+', ' ', base)

# ------------------ Path model ------------------
MODEL_DIR    = getattr(Config, "MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "models"))
MODEL_PATH   = os.path.join(MODEL_DIR, "final_model.pkl")
CLASSES_PATH = os.path.join(MODEL_DIR, "classes.json")  # fallback saja

class AIClassifier:
    def __init__(self):
        self._model = None
        self._classes = None   # urutan label (kebenaran utama dari clf.classes_)
        self._load()

    def _load(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model tak ditemukan: {MODEL_PATH}")

        # load cepat & hemat memori
        self._model = joblib.load(MODEL_PATH, mmap_mode='r')

        # Ambil urutan label dari classifier
        clf = None
        try:
            clf = self._model.named_steps['clf']
        except Exception:
            pass

        if clf is not None and hasattr(clf, 'classes_'):
            self._classes = list(clf.classes_)
            print(f"✅ Kelas diambil dari model.clf.classes_: {len(self._classes)} kelas")
        else:
            # fallback
            if os.path.exists(CLASSES_PATH):
                with open(CLASSES_PATH, "r", encoding="utf-8") as f:
                    self._classes = json.load(f)
                print(f"⚠️ clf.classes_ tidak ada — pakai classes.json ({len(self._classes)} kelas)")
            else:
                raise RuntimeError("Tidak menemukan daftar kelas (clf.classes_ / classes.json).")

        print(f"✅ Model dimuat: {MODEL_PATH}")

    def classify_document(self, raw_text: str, filename: str = ""):
        try:
            if not raw_text or not raw_text.strip():
                return {
                    "main_category": "Penunjang",
                    "sub_category": "Penunjang Lain",
                    "confidence": 0.0,
                    "predicted_label": "empty",
                    "top_results": []
                }

            toks_name = tokens_namafile(filename) if filename else ""
            teks_final = f"{raw_text} {toks_name}".strip()
            teks_bersih = pra_proses_teks(teks_final)

            proba = self._model.predict_proba([teks_bersih])[0]
            pred_idx = int(np.argmax(proba))
            label = self._classes[pred_idx]
            conf  = float(proba[pred_idx])

            if "_" in label:
                main_cat, sub_cat = label.split("_", 1)
            else:
                parts = re.split(r"[-\s]+", label, 1)
                main_cat = parts[0] if parts else "Penunjang"
                sub_cat  = parts[1] if len(parts) > 1 else "Penunjang Lain"

            # top-5 untuk diagnosa
            order = np.argsort(proba)[::-1]
            top5 = [{"label": self._classes[i], "prob": float(f"{proba[i]:.4f}")} for i in order[:5]]

            print(f"🧪 PRED: {label} | conf={conf:.4f} | len(text)={len(raw_text)} | file={filename} | top5={[(t['label'], t['prob']) for t in top5]}")

            return {
                "main_category": main_cat,
                "sub_category": sub_cat,
                "confidence": float(f"{conf:.4f}"),
                "predicted_label": label,
                "top_results": top5
            }

        except Exception as e:
            print(f"❌ classify_document error: {e}")
            return {
                "main_category": "Penunjang",
                "sub_category": "Penunjang Lain",
                "confidence": 0.0,
                "predicted_label": "error",
                "top_results": []
            }

# --------------- Lazy singleton ---------------
_ai_instance = None

def get_ai_classifier():
    global _ai_instance
    if _ai_instance is None:
        try:
            _ai_instance = AIClassifier()
        except Exception as e:
            print(f"⚠️ Gagal memuat AIClassifier: {e}")
            _ai_instance = None
    return _ai_instance
