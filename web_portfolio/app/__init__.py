# app/__init__.py
import os
from flask import Flask, redirect, url_for, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, current_user
from config import Config

db = SQLAlchemy()
login_manager = LoginManager()

def create_app():
    # __file__ ada di folder app/, jadi project root = satu level di atas
    PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))  # .../web_portfolio
    UPLOAD_DIR = os.path.join(PROJECT_ROOT, 'uploads')
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    app = Flask(
        __name__,
        template_folder=os.path.join(PROJECT_ROOT, 'templates'),
        static_folder=os.path.join(PROJECT_ROOT, 'static')
    )
    app.config.from_object(Config)

    # optional batas ukuran upload (misal 16MB)
    app.config.setdefault('MAX_CONTENT_LENGTH', 16 * 1024 * 1024)

    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'

    # ===== Blueprint routes =====
    from app.routes.auth import auth_bp
    from app.routes.profile import profile_bp
    from app.routes.portfolio import portfolio_bp
    from app.routes.services import services_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(profile_bp)
    app.register_blueprint(portfolio_bp)
    app.register_blueprint(services_bp)

    # ===== Home redirect =====
    @app.route('/')
    def index():
        if current_user.is_authenticated:
            return redirect(url_for('profile.dashboard'))
        else:
            return redirect(url_for('auth.login'))

    # ===== Static serving untuk file yang di-upload =====
    # Link yang dipakai di template: href="/uploads/<nama_file>"
    @app.route('/uploads/<path:filename>')
    def uploaded_files(filename):
        return send_from_directory(UPLOAD_DIR, filename, as_attachment=False)

    return app
