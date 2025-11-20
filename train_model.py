# train_model.py

# ============================================================
# 1) KONFIGURASI LINGKUNGAN & DEPENDENSI
# ============================================================

import os, io, re, warnings, sys

from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")                                      # gaya plot agar konsisten dan mudah dibaca

# --- Pustaka PDF & NLP ---
import PyPDF2                                                   # cepat untuk PDF berbasis teks
from pdfminer.high_level import extract_text_to_fp              # lebih akurat layout, tapi lebih lambat
from pdf2image import convert_from_path                         # render PDF -> image (untuk OCR)
import pytesseract                                              # OCR teks dari gambar
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tqdm import tqdm                                           # progress bar



# --- NLTK: tokenisasi & stopword ---

import nltk
from nltk.downloader import DownloadError

import nltk
from nltk.downloader import DownloadError

try:
nltk.data.find('tokenizers/punkt')
nltk.data.find('corpora/stopwords')
except DownloadError:
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, wordpunct_tokenize

# --- Sastrawi: stemmer Bahasa Indonesia ---

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
stemmer = StemmerFactory().create_stemmer()

# --- scikit-learn & imbalanced-learn: model, pipeline, evaluasi ---

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict, learning_curve, cross_val_score
from sklearn.metrics import (
classification_report, confusion_matrix, ConfusionMatrixDisplay,
roc_curve, auc, precision_recall_curve, average_precision_score,
accuracy_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.calibration import CalibratedClassifierCV

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import joblib, json, warnings
warnings.filterwarnings("ignore")  # menyembunyikan warning agar log bersih


# ============================================================
# 2) PARAMETER GLOBAL & STRUKTUR FOLDER
# ============================================================

ROOT_DIR = r'D:\Iya\dokumen_klasifikasi\data' 
MODEL_DIR = './models'
os.makedirs(MODEL_DIR, exist_ok=True)

SVD_COMPONENTS = 300                                            # Dimensi LSI. Umum: 100-300. Terlalu kecil: info hilang, terlalu besar: noise.
USE_OVERSAMPLING = True                                         # True: aktifkan SMOTE (disarankan untuk KNN saat data tak seimbang)
STOP_ID = set(stopwords.words('indonesian'))                    # daftar stopword Bahasa Indonesia

# ============================================================
# 3) EKSTRAKSI TEKS PDF (MULTI-TAHAP)
# ============================================================

def ekstrak_teks_pdf(pdf_path: str) -> str:
    teks_hasil = ""

    # 3.1) PyPDF2 (cepat, namun bisa gagal di PDF hasil scan)
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for halaman in reader.pages:
                t = halaman.extract_text()
                if t and t.strip():
                    teks_hasil += t + "\n"
        # Heuristik: jika teks hasil > 50 char, anggap sudah cukup baik
        if len(teks_hasil.strip()) > 50:
            return teks_hasil.strip()
    except Exception:
        pass                                                     # lanjut ke metode berikutnya

    # 3.2) pdfminer (lebih akurat untuk PDF kompleks, tapi lebih berat)
    try:
        with open(pdf_path, 'rb') as infile:
            out_str = io.StringIO()
            extract_text_to_fp(infile, out_str)
            t = out_str.getvalue().strip()
            if t:
                return t
    except Exception:
        pass                                                    # lanjut ke OCR

    # 3.3) OCR Tesseract (paling mahal; gunakan bahasa 'ind' untuk kualitas)
    try:
        images = convert_from_path(pdf_path)                    # render halaman menjadi gambar
        for img in images:
            teks_hasil += pytesseract.image_to_string(img, lang='ind')
    except Exception:
        pass                                                     # jika tetap gagal, kembalikan apapun yang ada (bisa kosong)

    return teks_hasil.strip()

# ============================================================
# 4) PRA-PEMROSESAN TEKS
# ============================================================

def pra_proses_teks(teks: str) -> str:
    teks = teks.lower()
    teks = re.sub(r'[^\w\s]', ' ', teks)                        # buang tanda baca
    teks = re.sub(r'\s+', ' ', teks).strip()                    # rapikan spasi
    try:
        tokens = word_tokenize(teks)                            # tokenizer utama (butuh 'punkt')
    except LookupError:
        tokens = wordpunct_tokenize(teks)                       # fallback jika resource belum ada
    # Hapus stopword dan token terlalu pendek
    tokens = [w for w in tokens if w not in STOP_ID and len(w) > 2]
    # Stemming: mengubah kata ke bentuk dasar
    return stemmer.stem(" ".join(tokens))

def tokens_namafile(path: str) -> str:
    """
    Kenapa menambahkan token dari nama file?
    - Banyak dokumen memberi sinyal kategori melalui nama file (mis. 'RPS_Pendidikan_2023').
    """
    base = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r'[_\-\.\$\$\$\$,]+', ' ', base)

# ============================================================
# 5) MUAT KORPUS DARI STRUKTUR FOLDER
# ============================================================

def load_corpus(root_dir):
    paths, labels = [], []
    if not os.path.exists(root_dir):
        print(f"[FATAL] ROOT_DIR not found: {root_dir}")
        sys.exit(1)

    print(f"Mencari dokumen di {root_dir}...")
    for main_cat in os.listdir(root_dir):
        main_path = os.path.join(root_dir, main_cat)
        if not os.path.isdir(main_path):
            continue
        for sub_cat in os.listdir(main_path):
            sub_path = os.path.join(main_path, sub_cat)
            if not os.path.isdir(sub_path):
                continue
            for fn in os.listdir(sub_path):
                if fn.lower().endswith('.pdf'):
                    paths.append(os.path.join(sub_path, fn))
                    labels.append(f"{main_cat}_{sub_cat}")      # label gabungan

    print(f"Ditemukan {len(paths)} dokumen. Ekstraksi dan Pra-pemrosesan Teks...")
    texts = [ekstrak_teks_pdf(p) for p in tqdm(paths, desc="Pemrosesan Dokumen", unit="it", leave=False)]

    corpus, y, skipped = [], [], []
    for t, lab, p in zip(texts, labels, paths):
        if t and t.strip():
            # Tambah sinyal dari nama file → sering membantu meningkatkan separabilitas
            teks_final = t + " " + tokens_namafile(p)
            corpus.append(pra_proses_teks(teks_final))
            y.append(lab)
        else:
            skipped.append(os.path.basename(p))

    # Log berkas yang dilewati (ekstraksi gagal/kosong)
    for fn in skipped:
        print(f"Lewati dokumen kosong: {fn}")

    # Buang kelas langka (1 sampel) → alasan: CV & metrik per-kelas tidak bermakna
    cnt = Counter(y)
    idx_keep = [i for i, lab in enumerate(y) if cnt[lab] > 1]
    corpus = [corpus[i] for i in idx_keep]
    y = [y[i] for i in idx_keep]

    if len(y) == 0:
        print("[FATAL] Korpus kosong setelah filter (atau semua kelas hanya punya 1 sampel).")
        sys.exit(1)

    # Statistik panjang token → indikasi kualitas ekstraksi/cleaning
    token_lengths = [len(c.split()) for c in corpus]
    print(f"\n{len(corpus)} dokumen siap dilatih ({len(set(y))} kategori)")
    return corpus, y, token_lengths

# ============================================================
# 6) DESKRIPSI DATA & VISUALISASI AWAL
# ============================================================

def deskripsi_dan_visualisasi(corpus, y, token_lengths):
    print("\n=============================================")
    print("| DESKRIPSI DATA & VISUALISASI AWAL |")
    print("=============================================")
    print(f"Jumlah Total Dokumen: {len(corpus)}")
    print(f"Jumlah Kategori: {len(set(y))}")

    freq = Counter(y)
    df_freq = pd.DataFrame.from_dict(freq, orient='index', columns=['Count']).sort_values('Count', ascending=False)
    print(df_freq.rename(index=lambda x: x.replace('_', ' ')))

    # Distribusi jumlah dokumen per kelas → indikasi imbalance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_freq.reset_index(), x='Count', y='index', palette='viridis')
    plt.title('Distribusi Kategori Dokumen (Data Asli)')
    plt.xlabel('Jumlah Dokumen'); plt.ylabel('Kategori')
    plt.tight_layout(); plt.savefig("1_Distribusi_Kategori.png"); plt.show()

    # Panjang token per dokumen → cek outlier (terlalu pendek = ekstraksi gagal)
    avg_tokens = np.mean(token_lengths)
    print(f"\nPanjang Token Rata-rata (Korpus): {avg_tokens:.2f}")

    plt.figure(figsize=(10, 6))
    plt.hist(token_lengths, bins=20, edgecolor='black', alpha=0.7)
    plt.title('Distribusi Jumlah Token per Dokumen')
    plt.xlabel('Jumlah Token'); plt.ylabel('Frekuensi')
    plt.axvline(avg_tokens, color='red', linestyle='--', label=f'Rata-rata: {avg_tokens:.2f}')
    plt.legend(); plt.tight_layout(); plt.savefig("2_Distribusi_Token.png"); plt.show()

# ============================================================
# 7) VISUALISASI EFEK SMOTE (OPSIONAL)
# ============================================================

def visualisasi_efek_smote(corpus, y, n_components=2):
    print("[INFO] Memvisualisasikan efek SMOTE (SVD 2D)...")
    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2)
    svd = TruncatedSVD(n_components=n_components, random_state=42)

    X_tfidf = tfidf.fit_transform(corpus)
    X_svd_before = svd.fit_transform(X_tfidf)

    # Catatan: Untuk teks, SMOTE bekerja di ruang fitur TF-IDF (bukan hasil SVD)
    smote = SMOTE(k_neighbors=1, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_tfidf, y)
    X_svd_after = svd.transform(X_resampled)

    classes = sorted(set(y))
    colors = plt.cm.Set1(np.linspace(0, 1, len(classes)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Sebelum SMOTE (imbalance)
    for i, cls in enumerate(classes):
        idx = [j for j, label in enumerate(y) if label == cls]
        ax1.scatter(X_svd_before[idx, 0], X_svd_before[idx, 1], c=[colors[i] for _ in idx], label=cls, alpha=0.6)
    ax1.set_title('Sebelum SMOTE (Imbalanced)'); ax1.legend()
    ax1.set_xlabel('SVD Komponen 1'); ax1.set_ylabel('SVD Komponen 2')

    # Sesudah SMOTE (balanced)
    for i, cls in enumerate(classes):
        idx = [j for j, label in enumerate(y_resampled) if label == cls]
        ax2.scatter(X_svd_after[idx, 0], X_svd_after[idx, 1], c=[colors[i] for _ in idx], label=cls, alpha=0.6)
    ax2.set_title('Sesudah SMOTE (Balanced)'); ax2.legend()
    ax2.set_xlabel('SVD Komponen 1'); ax2.set_ylabel('SVD Komponen 2')

    plt.tight_layout(); plt.savefig("3_Visualisasi_SMOTE.png"); plt.show()

# ============================================================
# 8) KLASIFIKASI: LSI + COSINE CENTROID
# ============================================================

class ClassCentroidCosineClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, use_softmax=True):
        self.use_softmax = use_softmax                          # jika True, kembalikan probabilitas "softmax" dari skor cosine

    @staticmethod
    def _toarray(X):
        # Banyak transformasi teks hasilkan sparse matrix; beberapa operasi butuh dense.
        return X.toarray() if hasattr(X, "toarray") else X

    def fit(self, X, y):
        X = self._toarray(X); y = np.array(y); self.classes_ = np.unique(y)
        cents = []
        for c in self.classes_:
            Xi = X[y == c]                                      # subset semua dokumen pada kelas c
            mu = Xi.mean(axis=0, keepdims=True)                 # centroid = rata-rata fitur
            denom = np.linalg.norm(mu)
            if denom > 0:                                       # normalisasi L2 agar cocok untuk cosine
                mu = mu / denom
            cents.append(mu)
        self.centroids_ = np.vstack(cents)                      # centroid per kelas
        return self

    def decision_function(self, X):
        X = self._toarray(X)
        # skor cosine: makin besar = makin dekat ke centroid kelas
        return cosine_similarity(X, self.centroids_)

    def predict(self, X):
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        # Softmax dari skor cosine (bukan probabilitas kalibrasi, namun cukup untuk ranking)
        s = self.decision_function(X)
        exp_s = np.exp(s - s.max(axis=1, keepdims=True))
        return exp_s / exp_s.sum(axis=1, keepdims=True)

# ============================================================
# 9) BUILDER PIPELINE LSI (+ OPSIONAL SMOTE)
# ============================================================

def make_lsi_pipeline(classifier, n_components=SVD_COMPONENTS, oversample=USE_OVERSAMPLING):
    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    norm = Normalizer(copy=False)

    # SMOTE bermanfaat untuk KNN saat kelas tidak seimbang (nearest neighbor sensitif pada densitas)
    if oversample and 'KNeighborsClassifier' in classifier.__class__.__name__:
        print("[INFO] Pipeline KNN menggunakan SMOTE(k_neighbors=1).")
        return ImbPipeline([
            ('tfidf', tfidf), ('svd', svd), ('norm', norm),
            ('smote', SMOTE(k_neighbors=1, random_state=42)),
            ('clf', classifier)
        ])
    elif 'ClassCentroidCosineClassifier' in classifier.__class__.__name__:
        print("[INFO] Pipeline Centroid Cosine: Tidak menggunakan SMOTE/sampling.")

    return Pipeline([('tfidf', tfidf), ('svd', svd), ('norm', norm), ('clf', classifier)])

# ============================================================
# 10) EVALUASI MODEL (CROSS-VALIDATION OOF) + GRAFIK
# ============================================================

def evaluate_and_plot(name, model, X, y, classes, cv):
    print("\n" + "=" * 56); print(f"| EVALUASI: {name} |"); print("=" * 56)
    file_name_prefix = name.replace(' ', '_').replace('+', '_')

    # Prediksi out-of-fold agar fair (tiap fold jadi "data uji")
    y_pred_cv = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
    acc = accuracy_score(y, y_pred_cv)
    print(f"Akurasi (CV): {acc:.4f}")
    print(classification_report(y, y_pred_cv, zero_division=0))

    rep = classification_report(y, y_pred_cv, output_dict=True, zero_division=0)
    f1_w = rep.get('weighted avg', {}).get('f1-score', np.nan)

    # Confusion Matrix → melihat pola salah klasifikasi dominan
    cm = confusion_matrix(y, y_pred_cv, labels=classes)
    plt.figure(figsize=(10, 8))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot(
        cmap=plt.cm.Blues, xticks_rotation='vertical', values_format='d'
    )
    plt.title(f'Confusion Matrix (CV) — {name}')
    plt.tight_layout(); plt.savefig(f"4_CM_{file_name_prefix}.png"); plt.show()

    # ROC & PR hanya tersedia jika model punya predict_proba
    auc_macro = np.nan; ap_macro = np.nan
    try:
        y_proba_cv = cross_val_predict(model, X, y, cv=cv, method='predict_proba', n_jobs=-1)
        y_bin = label_binarize(y, classes=classes)

        plt.figure(figsize=(12, 4))
        # ROC (OVR)
        ax1 = plt.subplot(1, 2, 1); aucs = []
        for i, cls in enumerate(classes):
            if y_bin[:, i].sum() == 0: 
                continue  # lewati jika kelas terlalu jarang
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba_cv[:, i])
            roc_auc = auc(fpr, tpr); aucs.append(roc_auc)
            ax1.plot(fpr, tpr, lw=1.8, label=f"{cls} (AUC={roc_auc:.2f})")
        ax1.plot([0, 1], [0, 1], 'k--', lw=1)
        ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR"); ax1.set_title(f"ROC — {name} (CV)")
        ax1.legend(fontsize=8, loc="lower right")

        # Precision–Recall (OVR)
        ax2 = plt.subplot(1, 2, 2); aps = []
        for i, cls in enumerate(classes):
            if y_bin[:, i].sum() == 0:
                continue
            p, r, _ = precision_recall_curve(y_bin[:, i], y_proba_cv[:, i])
            ap = average_precision_score(y_bin[:, i], y_proba_cv[:, i]); aps.append(ap)
            ax2.plot(r, p, lw=1.8, label=f"{cls} (AP={ap:.2f})")
        ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision"); ax2.set_title(f"Precision–Recall — {name} (CV)")
        ax2.legend(fontsize=8, loc="lower left")

        plt.tight_layout(); plt.savefig(f"5_ROC_PR_{file_name_prefix}.png"); plt.show()

        auc_macro = float(np.mean(aucs)) if aucs else np.nan
        ap_macro  = float(np.mean(aps))  if aps  else np.nan
    except Exception as e:
        print(f"[WARN] ROC/PR dilewati untuk {name}: {e!r}  (kemungkinan model tidak mendukung probabilitas)")

    return acc, f1_w, auc_macro, ap_macro

# ============================================================
# 11) MAIN EXECUTION
# ============================================================

if __name__ == '__main__':
    # (a) Muat & siapkan data
    corpus_main, labels, token_lengths = load_corpus(ROOT_DIR)
    classes = sorted(set(labels))

    # (b) Deskripsi & visualisasi awal
    deskripsi_dan_visualisasi(corpus_main, labels, token_lengths)

    # (c) Visualisasi efek SMOTE (hanya untuk penjelasan; training utama tetap di CV)
    if USE_OVERSAMPLING:
        visualisasi_efek_smote(corpus_main, labels)

    # (d) Siapkan skema CV
    # - Batas fold mengikuti jumlah sampel minimum per kelas (agar setiap fold punya data kelas tsb).
    min_class = min(Counter(labels).values())
    cv_folds = max(2, min(5, min_class))  # aman 2–5 fold
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # (e) Siapkan dua kandidat model
    pipe_cos = make_lsi_pipeline(ClassCentroidCosineClassifier(use_softmax=True))       # LSI + Cosine
    knn_base = KNeighborsClassifier(n_neighbors=5, weights='distance', p=2)             # KNN baseline
    pipe_knn_base = make_lsi_pipeline(knn_base, oversample=USE_OVERSAMPLING)            # LSI + KNN (+SMOTE jika aktif)

    print(f"\n--- Konfigurasi Akhir ---")
    print(f"LSI Components: {SVD_COMPONENTS}")
    print(f"SMOTE: {USE_OVERSAMPLING}")
    print(f"CV Folds: {cv_folds} (min class={min_class})")
    print("-------------------------\n")

    # (f) Grid Search KNN
    # - Tujuan: menemukan K, bobot tetangga, dan metrik jarak terbaik (p=1 Manhattan, p=2 Euclidean)
    param_grid = {
        'clf__n_neighbors': [3, 5, 7, 9, 11],
        'clf__weights': ['uniform', 'distance'],
        'clf__p': [1, 2],
    }
    print(f"=== GridSearch LSI+KNN (CV={cv_folds}) ===")
    gs_knn = GridSearchCV(pipe_knn_base, param_grid=param_grid, cv=cv,
                          scoring='f1_weighted', n_jobs=-1, verbose=1)
    gs_knn.fit(corpus_main, labels)
    best_model_knn = gs_knn.best_estimator_
    print(f"Best params (KNN): {gs_knn.best_params_}")
    print(f"Best CV (F1-weighted): {gs_knn.best_score_:.4f}")

    # (g) Bandingkan kinerja LSI+Cosine vs LSI+KNN(best)
    models = {
        "LSI+Cosine Similarity (Centroid)": pipe_cos,
        f"LSI+KNN (K={best_model_knn.named_steps['clf'].n_neighbors})": best_model_knn
    }
    summary_df = pd.DataFrame(index=models.keys(),
                              columns=['Accuracy','F1-Weighted','AUC-Macro','AP-Macro'])
    for name, model in models.items():
        acc, f1w, aucm, apm = evaluate_and_plot(name, model, corpus_main, labels, classes, cv)
        summary_df.loc[name] = [acc, f1w, aucm, apm]

    print("\n\n========================================================")
    print("| RINGKASAN HASIL EVALUASI KOMPARATIF (CV) |")
    print("========================================================")
    print(summary_df)

    # (h) Plot ringkasan perbandingan metrik utama agar mudah dipresentasikan
    metrics = ['Accuracy', 'F1-Weighted', 'AUC-Macro', 'AP-Macro']
    x = np.arange(len(metrics)); width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    row0 = pd.to_numeric(summary_df.iloc[0][metrics], errors='coerce').astype(float).values
    row1 = pd.to_numeric(summary_df.iloc[1][metrics], errors='coerce').astype(float).values
    ax.bar(x - width/2, row0, width, label=summary_df.index[0])
    ax.bar(x + width/2, row1, width, label=summary_df.index[1])
    ax.set_ylabel('Skor'); ax.set_title('Perbandingan Kinerja (LSI+Cosine vs LSI+KNN)')
    ax.set_xticks(x); ax.set_xticklabels(metrics); ax.legend(); ax.grid(axis='y')
    yall = np.concatenate([row0, row1])
    ymin = max(0.0, np.nanmin(yall)*0.9) if np.nanmin(yall) >= 0 else 0.0
    plt.ylim(ymin, 1.05); plt.tight_layout(); plt.savefig("6_Perbandingan_Kinerja.png"); plt.show()

    # (i) Analisis perilaku model (opsional tapi kuat untuk bab pembahasan)
    #     Termasuk: overfitting curve, learning curve, dan reliability (kalibrasi proba).
    pipe_knn_no_sampling = make_lsi_pipeline(
        KNeighborsClassifier(n_neighbors=5, weights='distance', p=2),
        n_components=SVD_COMPONENTS, oversample=False
    )
    pipe_cos_no_sampling = make_lsi_pipeline(
        ClassCentroidCosineClassifier(use_softmax=True),
        n_components=SVD_COMPONENTS, oversample=False
    )

    # FIX: DEFINISIKAN train_sizes DI SINI SEBELUM DIGUNAKAN
    train_sizes = np.linspace(0.1, 1.0, 6) # <--- TAMBAHKAN/PINDAHKAN BARIS INI
    
    # Overfitting curve (Cosine vs dimensi SVD)
    components_list = [100, 150, 200, 250, 300]
    train_acc_cos, cv_acc_cos = [], []
    print("\n[INFO] Menghitung Kurva Overfitting Cosine...") # <--- PESAN DIMINTA
    
    # MENGHILANGKAN TQDM (Mencegah progress bar muncul)
    for n_comp in components_list: 
        # Catatan: Pesan "[INFO] Pipeline Centroid Cosine: Tidak menggunakan SMOTE/sampling." 
        # dari make_lsi_pipeline akan tetap muncul 5 kali.
        mdl_cos = make_lsi_pipeline(ClassCentroidCosineClassifier(use_softmax=True),
                                     n_components=n_comp, oversample=False)
        mdl_cos.fit(corpus_main, labels)
        train_acc_cos.append(accuracy_score(labels, mdl_cos.predict(corpus_main)))
        cv_acc_cos.append(cross_val_score(mdl_cos, corpus_main, labels, cv=cv, scoring='accuracy', n_jobs=-1).mean())
    plt.figure(figsize=(10, 6))
    plt.plot(components_list, train_acc_cos, marker='o', label='Train Acc')
    plt.plot(components_list, cv_acc_cos, marker='o', label='CV Acc')
    plt.title('Overfitting Curve — LSI+Cosine')
    plt.xlabel('Jumlah Komponen SVD (LSI)'); plt.ylabel('Akurasi')
    plt.grid(True); plt.legend(); plt.savefig("7_Overfitting_Cosine.png"); plt.show()

    # Learning curve (Cosine)
    print("[INFO] Menghitung Learning Curve Cosine...") # <--- PESAN DIMINTA
    tr_sizes, tr_scores, cv_scores = learning_curve(estimator=pipe_cos_no_sampling,
                                                     X=corpus_main, y=labels,
                                                     train_sizes=train_sizes, cv=cv,
                                                     scoring='accuracy', n_jobs=-1)
    plt.figure(figsize=(10, 6))
    plt.plot(tr_sizes, tr_scores.mean(axis=1), marker='o', label='Train accuracy')
    plt.plot(tr_sizes, cv_scores.mean(axis=1), marker='o', label='CV accuracy')
    plt.title('LSI+Cosine Learning Curve (accuracy)')
    plt.xlabel('Jumlah data latih'); plt.ylabel('Accuracy'); plt.grid(True); plt.legend()
    plt.savefig("8_Learning_Curve_Acc_Cosine.png"); plt.show()

    # Reliability (Cosine) — menggunakan "probabilitas" softmax dari skor cosine
    print("[INFO] Menghitung Reliability Curve Cosine...") # <--- PESAN DIMINTA
    try:
        y_proba_cal_cos = cross_val_predict(pipe_cos_no_sampling, corpus_main, labels, cv=cv,
                                             method='predict_proba', n_jobs=-1)
        y_pred_cos = np.argmax(y_proba_cal_cos, axis=1)
        labels_arr = np.array([classes.index(c) for c in labels])
        max_proba_cos = y_proba_cal_cos[np.arange(len(y_proba_cal_cos)), y_pred_cos]
        correct_cos = (y_pred_cos == labels_arr).astype(int)

        bins = np.linspace(0.0, 1.0, 8); bin_ids = np.digitize(max_proba_cos, bins) - 1
        x_bin_cos, y_bin_cos = [], []
        for b in range(len(bins)-1):
            idx = (bin_ids == b)
            if idx.sum() == 0: continue
            x_bin_cos.append(max_proba_cos[idx].mean()); y_bin_cos.append(correct_cos[idx].mean())

        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        plt.plot(x_bin_cos, y_bin_cos, marker='o', label='Raw predict_proba (Cosine)')
        plt.xlabel('Predicted probability (max class)'); plt.ylabel('Empirical accuracy')
        plt.title('Reliability Curve — LSI+Cosine'); plt.legend(); plt.grid(True)
        plt.savefig("9_Reliability_Cosine.png"); plt.show()
        
        # BARIS INI DIHAPUS agar output lebih ringkas:
        # print("[INFO] Reliability Curve untuk LSI+Cosine berhasil diplot (ingat: ini softmax skor cosine, bukan kalibrasi Platt).") 
        
    except Exception as e:
        print(f"[WARN] Reliability Curve untuk LSI+Cosine dilewati: {e!r}")
        
    # Overfitting curve (KNN vs K)
    ks = [1,2,3,4,5,7,9,11]; train_acc, cv_acc = [], []
    print("\n[INFO] Menghitung Kurva Overfitting KNN...")
    for k in tqdm(ks, desc="KNN Overfitting", leave=False):
        mdl = clone(pipe_knn_no_sampling); mdl.named_steps['clf'].n_neighbors = k
        mdl.fit(corpus_main, labels)
        train_acc.append(accuracy_score(labels, mdl.predict(corpus_main)))              # jika jauh > CV, indikasi overfit
        cv_acc.append(cross_val_score(mdl, corpus_main, labels, cv=cv, scoring='accuracy', n_jobs=-1).mean())
    plt.figure(figsize=(10, 6))
    plt.plot(ks, train_acc, marker='o', label='Train Acc')
    plt.plot(ks, cv_acc, marker='o', label='CV Acc')
    plt.title('Overfitting Curve — LSI+KNN'); plt.xlabel('Jumlah Tetangga (K)'); plt.ylabel('Akurasi')
    plt.grid(True); plt.legend(); plt.savefig("7_Overfitting_KNN.png"); plt.show()

    # Learning curve (KNN)
    print("[INFO] Menghitung Learning Curve KNN...")
    tr_sizes, tr_scores, cv_scores = learning_curve(estimator=pipe_knn_no_sampling,
                                                    X=corpus_main, y=labels,
                                                    train_sizes=train_sizes, cv=cv,
                                                    scoring='accuracy', n_jobs=-1)
    plt.figure(figsize=(10, 6))
    plt.plot(tr_sizes, tr_scores.mean(axis=1), marker='o', label='Train accuracy')
    plt.plot(tr_sizes, cv_scores.mean(axis=1), marker='o', label='CV accuracy')
    plt.title('LSI+KNN Learning Curve (accuracy)')
    plt.xlabel('Jumlah data latih'); plt.ylabel('Accuracy'); plt.grid(True); plt.legend()
    plt.savefig("8_Learning_Curve_Acc_KNN.png"); plt.show()

    # Reliability / Calibration (KNN)
    print("[INFO] Menghitung Reliability Curve KNN...")
    try:
        calib = CalibratedClassifierCV(estimator=pipe_knn_no_sampling, method='sigmoid', cv=2)
        y_proba_cal = cross_val_predict(calib, corpus_main, labels, cv=cv, method='predict_proba', n_jobs=-1)
        y_pred = np.argmax(y_proba_cal, axis=1)
        labels_arr = np.array([classes.index(c) for c in labels])
        max_proba = y_proba_cal[np.arange(len(y_proba_cal)), y_pred]
        correct = (y_pred == labels_arr).astype(int)

        # Bin probabilitas → titik di kurva mendekati diagonal berarti kalibrasi baik
        bins = np.linspace(0.0, 1.0, 8); bin_ids = np.digitize(max_proba, bins) - 1
        x_bin, y_bin = [], []
        for b in range(len(bins)-1):
            idx = (bin_ids == b)
            if idx.sum() == 0: continue
            x_bin.append(max_proba[idx].mean())
            y_bin.append(correct[idx].mean())

        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        plt.plot(x_bin, y_bin, marker='o', label='Calibrated (sigmoid, cv=2)')
        plt.xlabel('Predicted probability (max class)'); plt.ylabel('Empirical accuracy')
        plt.title('Reliability Curve — LSI+KNN'); plt.legend(); plt.grid(True)
        plt.savefig("9_Reliability_KNN.png"); plt.show()
    except Exception as e:
        print(f"[WARN] Reliability Curve untuk KNN dilewati: {e!r}")

    # (j) Pilih model terbaik berdasarkan F1-Weighted (lebih adil saat data tak seimbang)
    best_model_name = summary_df['F1-Weighted'].astype(float).idxmax()
    best_final_model = clone(models[best_model_name]).fit(corpus_main, labels)
    print(f"\nModel TERBAIK ({best_model_name}) dilatih ulang di seluruh data dan siap digunakan.")

    # (k) Simpan artefak untuk integrasi aplikasi (Flask, dsb.)
    model_path = os.path.join(MODEL_DIR, 'final_model.pkl')
    classes_path = os.path.join(MODEL_DIR, 'classes.json')
    joblib.dump(best_final_model, model_path)                   # pipeline lengkap (TF-IDF → SVD → Normalizer → (SMOTE) → clf)
    with open(classes_path, 'w') as f:
        json.dump(classes, f)                                   # urutan kelas untuk mapping indeks → label
    print(f"[SUKSES] Model terbaik disimpan ke: {model_path}")
    print(f"[SUKSES] Kelas disimpan ke: {classes_path}")
