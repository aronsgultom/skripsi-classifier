from flask import Blueprint, render_template, request, jsonify, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import os

from app import db
from app.models import Portfolio

portfolio_bp = Blueprint('portfolio', __name__)

# =========================
# Kategori (sesuaikan bila perlu)
# =========================
CATEGORIES = {
    'Penelitian': ['Paten', 'Penelitian', 'Publikasi Karya'],
    'Pendidikan': [
        'Bahan Ajar', 'Bimbingan Mahasiswa', 'Datasering', 'Orasi Ilmiah',
        'Pembimbing Dosen', 'Pembinaan Mahasiswa', 'Pengajaran',
        'Pengujian Mahasiswa', 'Tugas Tambahan', 'Visiting Scientist'
    ],
    'Pengabdian': ['Jabatan Struktural', 'Pembicara', 'Pengabdian', 'Pengelola Jurnal'],
    'Penunjang': ['Anggota Profesi', 'Penghargaan', 'Penunjang Lain']
}

# =========================
# Upload config
# =========================
ALLOWED_EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg'}
UPLOAD_DIR_NAME = 'uploads'  # disimpan di root project: <project>/uploads

def _ensure_upload_folder():
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # .../web_portfolio
    upload_dir = os.path.join(root_dir, UPLOAD_DIR_NAME)
    os.makedirs(upload_dir, exist_ok=True)
    return upload_dir

def _allowed_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS

# =========================
# Routes
# =========================
@portfolio_bp.route('/portfolio')
@login_required
def index():
    portfolios = Portfolio.query.filter_by(user_id=current_user.id).order_by(Portfolio.created_at.desc()).all()
    return render_template('portfolio.html', portfolios=portfolios, categories=CATEGORIES)

@portfolio_bp.route('/portfolio/add', methods=['POST'])
@login_required
def add_portfolio():
    # Ambil form fields
    main_category = request.form.get('main_category')
    sub_category  = request.form.get('sub_category')
    title         = request.form.get('title')

    # Validasi dasar
    if not main_category or not sub_category or not title:
        return jsonify({'success': False, 'message': 'Main Category, Sub Category, dan Title wajib diisi.'}), 400

    # File wajib
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'File wajib diunggah.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'Pilih file terlebih dahulu.'}), 400

    if not _allowed_file(file.filename):
        return jsonify({'success': False, 'message': 'Format file tidak didukung (hanya .pdf/.png/.jpg/.jpeg).'}), 400

    # Simpan file
    upload_dir = _ensure_upload_folder()
    filename = secure_filename(file.filename)
    save_path = os.path.join(upload_dir, filename)

    # Hindari overwrite—beri suffix angka jika nama sudah ada
    base, ext = os.path.splitext(filename)
    i = 1
    while os.path.exists(save_path):
        filename = f"{base}({i}){ext}"
        save_path = os.path.join(upload_dir, filename)
        i += 1

    try:
        file.save(save_path)
        # Simpan ke DB (path relatif untuk dipakai di link: uploads/filename)
        p = Portfolio(
            user_id=current_user.id,
            main_category=main_category,
            sub_category=sub_category,
            title=title,
            file_path=f"{UPLOAD_DIR_NAME}/{filename}",
            description=None  # kita tidak pakai description lagi
        )
        db.session.add(p)
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': f'Gagal menyimpan file: {e}'}), 500

@portfolio_bp.route('/portfolio/<int:id>', methods=['GET', 'PUT', 'DELETE'])
@login_required
def manage_portfolio(id):
    p = Portfolio.query.filter_by(id=id, user_id=current_user.id).first_or_404()

    # ---- GET: kembalikan data untuk modal edit
    if request.method == 'GET':
        data = {
            'id': p.id,
            'main_category': p.main_category,
            'sub_category': p.sub_category,
            'title': p.title,
            'file_path': p.file_path
        }
        return jsonify({'success': True, 'portfolio': data})

    # ---- PUT: update field + opsional ganti file baru
    if request.method == 'PUT':
        main_category = request.form.get('main_category') or p.main_category
        sub_category  = request.form.get('sub_category') or p.sub_category
        title         = request.form.get('title') or p.title

        # Optional file replace
        new_file = request.files.get('file')
        upload_dir = _ensure_upload_folder()

        try:
            if new_file and new_file.filename:
                if not _allowed_file(new_file.filename):
                    return jsonify({'success': False, 'message': 'Format file tidak didukung.'}), 400

                filename = secure_filename(new_file.filename)
                save_path = os.path.join(upload_dir, filename)

                base, ext = os.path.splitext(filename)
                i = 1
                while os.path.exists(save_path):
                    filename = f"{base}({i}){ext}"
                    save_path = os.path.join(upload_dir, filename)
                    i += 1

                new_file.save(save_path)

                # Hapus file lama (jika ada & berbeda)
                if p.file_path:
                    old_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), p.file_path)
                    try:
                        if os.path.exists(old_path):
                            os.remove(old_path)
                    except OSError:
                        pass

                p.file_path = f"{UPLOAD_DIR_NAME}/{filename}"

            # Update meta
            p.main_category = main_category
            p.sub_category  = sub_category
            p.title         = title

            db.session.commit()
            return jsonify({'success': True})
        except Exception as e:
            db.session.rollback()
            return jsonify({'success': False, 'message': f'Gagal update: {e}'}), 500

    # ---- DELETE: hapus record + file fisik
    if request.method == 'DELETE':
        try:
            if p.file_path:
                old_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), p.file_path)
                try:
                    if os.path.exists(old_path):
                        os.remove(old_path)
                except OSError:
                    pass
            db.session.delete(p)
            db.session.commit()
            return jsonify({'success': True})
        except Exception as e:
            db.session.rollback()
            return jsonify({'success': False, 'message': f'Gagal menghapus: {e}'}), 500
