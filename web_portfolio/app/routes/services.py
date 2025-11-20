from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import os, io
from app import db
from app.models import Document, Portfolio

# Import classifier (lazy singleton)
from app.ml.classifier import get_ai_classifier

# === DEPENDENSI untuk ekstraksi PDF ===
import PyPDF2
from pdfminer.high_level import extract_text_to_fp
from pdf2image import convert_from_path
import pytesseract

# ============================================================
# Fungsi Ekstraksi PDF — sama persis seperti di train_model.py
# ============================================================
def ekstrak_teks_pdf(pdf_path):
    """Mencoba berbagai metode (PyPDF2 → pdfminer → OCR) untuk ekstraksi teks PDF."""
    teks_hasil = ""

    # 1️⃣ PyPDF2 — cepat, tapi tidak selalu lengkap
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for halaman in reader.pages:
                t = halaman.extract_text()
                if t and t.strip():
                    teks_hasil += t + "\n"
        if len(teks_hasil.strip()) > 50:
            return teks_hasil.strip()
    except Exception as e:
        print(f"❌ PyPDF2 gagal: {e}")

    # 2️⃣ pdfminer.six — lebih dalam, tapi agak lambat
    try:
        with open(pdf_path, 'rb') as infile:
            out_str = io.StringIO()
            extract_text_to_fp(infile, out_str)
            t = out_str.getvalue().strip()
            if t:
                return t
    except Exception as e:
        print(f"⚠️ pdfminer gagal: {e}")

    # 3️⃣ OCR — fallback terakhir, pakai Tesseract (harus sudah diinstal)
    try:
        images = convert_from_path(pdf_path)
        for img in images:
            teks_hasil += pytesseract.image_to_string(img, lang='ind')
    except Exception as e:
        print(f"⚠️ OCR gagal: {e}")

    return teks_hasil.strip()

# ============================================================
# ROUTES
# ============================================================
services_bp = Blueprint('services', __name__)

ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@services_bp.route('/')
@services_bp.route('/index')
@login_required
def index():
    return redirect(url_for('services.services'))


@services_bp.route('/services')
@login_required
def services():
    documents = Document.query.filter_by(user_id=current_user.id).all()
    return render_template('services.html', documents=documents)


@services_bp.route('/upload', methods=['POST'])
@login_required
def upload():
    try:
        if 'file' not in request.files:
            flash('No file uploaded', 'error')
            return redirect(url_for('services.services'))

        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('services.services'))

        if not allowed_file(file.filename):
            flash('File harus berformat PDF.', 'error')
            return redirect(url_for('services.services'))

        # === Simpan file ke folder uploads ===
        filename = secure_filename(file.filename)
        upload_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'uploads')
        os.makedirs(upload_folder, exist_ok=True)

        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)

        # === Ekstraksi teks pakai fungsi yang sama dengan training ===
        extracted_text = ""
        try:
            extracted_text = ekstrak_teks_pdf(file_path) or ""
        except Exception as e:
            print(f"❌ ekstrak_teks_pdf error: {e}")
            extracted_text = os.path.splitext(filename)[0]

        print(f"📝 Extracted length: {len(extracted_text)} | file={filename}")

        # === Ambil classifier (lazy load) ===
        clf = get_ai_classifier()
        if clf is None:
            flash('Model belum tersedia, tidak bisa klasifikasi saat ini.', 'error')
            return redirect(url_for('services.services'))

        # === Klasifikasi dokumen ===
        result = clf.classify_document(extracted_text, filename=filename)
        confidence   = float(result.get('confidence', 0.0))
        main_category = result.get('main_category', 'Penunjang')
        sub_category  = result.get('sub_category',  'Penunjang Lain')

        # Jika confidence rendah & hasilnya Penunjang Lain → tandai Uncertain
        if confidence < 0.55 and (main_category == 'Penunjang' and sub_category == 'Penunjang Lain'):
            print("⚠️ Low confidence on Penunjang Lain — marking as Uncertain")
            main_category = 'Uncertain'
            sub_category  = 'Needs Review'

        # === Simpan hasil ke database ===
        document = Document(
            user_id=current_user.id,
            file_name=filename,
            file_path=f'uploads/{filename}',
            predicted_category=f"{main_category} - {sub_category}"
        )
        db.session.add(document)

        portfolio = Portfolio(
            user_id=current_user.id,
            main_category=main_category,
            sub_category=sub_category,
            title=f"Diklasifikasikan Otomatis: {os.path.splitext(filename)[0]}",
            description=f"Dokumen diklasifikasikan otomatis sebagai {main_category} - {sub_category} "
                        f"(confidence {confidence:.1%})",
            file_path=f'uploads/{filename}'
        )
        db.session.add(portfolio)
        db.session.commit()

        flash(f'Document uploaded and classified as {main_category} - {sub_category}', 'success')
        return redirect(url_for('services.services'))

    except Exception as e:
        db.session.rollback()
        print(f"❌ Upload error: {e}")
        flash('Error uploading document', 'error')
        return redirect(url_for('services.services'))


@services_bp.route('/delete_document/<int:document_id>', methods=['DELETE'])
@login_required
def delete_document(document_id):
    """Hapus dokumen + portofolio terkait"""
    try:
        document = Document.query.filter_by(id=document_id, user_id=current_user.id).first()
        if not document:
            return jsonify({'success': False, 'message': 'Dokumen tidak ditemukan'}), 404

        # Hapus portofolio terkait
        associated = Portfolio.query.filter_by(
            user_id=current_user.id, file_path=document.file_path
        ).all()
        for p in associated:
            db.session.delete(p)

        # Hapus file fisik
        file_full_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), document.file_path)
        if os.path.exists(file_full_path):
            try:
                os.remove(file_full_path)
            except OSError:
                pass

        db.session.delete(document)
        db.session.commit()
        return jsonify({'success': True, 'message': 'Dokumen berhasil dihapus'})

    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': f'Gagal menghapus dokumen: {str(e)}'}), 500
