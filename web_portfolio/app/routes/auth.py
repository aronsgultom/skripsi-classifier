from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required, current_user
from app import db
from app.models import User, Profile
from werkzeug.security import generate_password_hash

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('profile.dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        print(f"🔍 Login attempt - Username: '{username}', Password length: {len(password)}")
        
        if not username or not password:
            flash('Please enter both username and password', 'error')
            return render_template('login.html')
        
        user = User.query.filter_by(username=username).first()
        
        if user:
            print(f"✅ User found: {user.username} (ID: {user.id})")
            print(f"🔐 Password hash exists: {'Yes' if user.password_hash else 'No'}")
            
            if user.check_password(password):
                print("🎉 Password correct, logging in...")
                login_user(user)
                flash(f'Welcome back, {user.username}!', 'success')
                return redirect(url_for('profile.dashboard'))
            else:
                print("❌ Password incorrect")
                flash('Invalid username or password. Please check your credentials and try again.', 'error')
        else:
            print(f"❌ User '{username}' not found")
            flash('Invalid username or password. Please check your credentials and try again.', 'error')
            
    return render_template('login.html')

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('profile.dashboard'))

    if request.method == 'POST':
        try:
            username = request.form.get('username').strip()
            email = request.form.get('email').strip()
            password = request.form.get('password')
            
            print(f"Registration attempt: {username}, {email}")  # Debug log
            
            # Validate input
            if not username or not email or not password:
                flash('All fields are required', 'error')
                return render_template('register.html')
                
            if len(password) < 6:
                flash('Password must be at least 6 characters long', 'error')
                return render_template('register.html')
            
            # Check if user already exists
            existing_user = User.query.filter_by(username=username).first()
            if existing_user:
                print(f"Username {username} already exists")  # Debug log
                flash('Username already exists. Please choose a different username.', 'error')
                return render_template('register.html')
            
            existing_email = User.query.filter_by(email=email).first()
            if existing_email:
                print(f"Email {email} already registered")  # Debug log
                flash('Email already registered. Please use a different email or login.', 'error')
                return render_template('register.html')
            
            # Create new user
            user = User(username=username, email=email)
            user.set_password(password)
            
            print(f"Creating user: {username}")  # Debug log
            print(f"Password hash: {user.password_hash[:20]}...")  # Debug log (partial)
            
            db.session.add(user)
            db.session.flush()  # Get user ID without committing
            
            # Create empty profile for user
            profile = Profile(user_id=user.id)
            db.session.add(profile)
            
            db.session.commit()
            
            print(f"User {username} created successfully with ID: {user.id}")  # Debug log
            
            flash('Registration successful! You can now login with your credentials.', 'success')
            return redirect(url_for('auth.login'))
            
        except Exception as e:
            db.session.rollback()
            print(f"Registration error: {e}")  # Debug log
            flash('Registration failed. Please try again or contact support.', 'error')
            return render_template('register.html')
            
    return render_template('register.html')

@auth_bp.route('/logout')
@login_required
def logout():
    username = current_user.username
    logout_user()
    flash(f'You have been logged out successfully, {username}. See you next time!', 'success')
    return redirect(url_for('auth.login'))