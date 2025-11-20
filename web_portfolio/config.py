import os

class Config:
    SECRET_KEY = 'your-secret-key-here'
    
    # Database configuration
    SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://root:@localhost/portfolio_db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Upload configuration
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

    # Model configuration
    # === GUNAKAN PATH ABSOLUT KE FOLDER MODEL ===
    MODEL_DIR = r"D:\Iya\dokumen_klasifikasi\models"
