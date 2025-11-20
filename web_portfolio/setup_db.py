from app import create_app, db

app = create_app()

print("🔧 Setting up database connection...")
print(f"Database URI: {app.config['SQLALCHEMY_DATABASE_URI']}")

with app.app_context():
    try:
        # Test database connection
        db.engine.connect()
        print("✅ Database connection successful!")
        
        # Create all tables
        db.create_all()
        print("✅ Database tables created successfully!")
        print("📋 Tables created:")
        print("   - user")
        print("   - profile") 
        print("   - portfolio")
        print("   - document")
        
    except Exception as e:
        print(f"❌ Error setting up database: {e}")
        print("\n🛠️ Troubleshooting:")
        print("1. Make sure XAMPP is running")
        print("2. Make sure MySQL service is started")
        print("3. Make sure database 'portfolio_db' exists in phpMyAdmin")
        print("4. Install PyMySQL: pip install PyMySQL")