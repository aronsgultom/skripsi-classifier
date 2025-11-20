from app import create_app, db
from app.models import User, Profile

app = create_app()

with app.app_context():
    print("🔧 Fixing admin user password...")
    
    # Delete existing admin user
    admin_user = User.query.filter_by(username='admin').first()
    if admin_user:
        db.session.delete(admin_user)
        db.session.commit()
        print("🗑️ Deleted old admin user")
    
    # Create fresh admin user
    admin_user = User(username='admin', email='admin@admin.com')
    admin_user.set_password('123456')
    
    print(f"🔐 New password hash: {admin_user.password_hash}")
    
    db.session.add(admin_user)
    db.session.flush()  # Get user ID
    
    # Create profile for admin
    admin_profile = Profile(user_id=admin_user.id)
    db.session.add(admin_profile)
    
    db.session.commit()
    
    # Test immediately
    test_result = admin_user.check_password('123456')
    print(f"✅ Password test: {'PASS' if test_result else 'FAIL'}")
    
    if test_result:
        print("🎉 Admin user fixed! You can now login with:")
        print("   Username: admin")
        print("   Password: 123456")
        print(f"   Profile created with ID: {admin_profile.id}")
    else:
        print("❌ Still having issues with password...")
        
        # Try manual verification
        from werkzeug.security import check_password_hash
        manual_test = check_password_hash(admin_user.password_hash, '123456')
        print(f"🔍 Manual test: {'PASS' if manual_test else 'FAIL'}")