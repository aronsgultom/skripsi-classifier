from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify
from flask_login import login_required, current_user
from app import db
from app.models import Profile, Portfolio, Document

profile_bp = Blueprint('profile', __name__)

@profile_bp.route('/dashboard')
@login_required
def dashboard():
    # Get user statistics - real-time counts
    portfolio_count = Portfolio.query.filter_by(user_id=current_user.id).count()
    document_count = Document.query.filter_by(user_id=current_user.id).count()
    
    return render_template('dashboard.html', 
                         user=current_user,
                         portfolio_count=portfolio_count,
                         document_count=document_count)

@profile_bp.route('/profile', methods=['GET', 'POST'])
@login_required
def manage_profile():
    profile = Profile.query.filter_by(user_id=current_user.id).first()
    if not profile:
        profile = Profile(user_id=current_user.id)
        db.session.add(profile)
        db.session.commit()

    if request.method == 'POST':
        profile.full_name = request.form.get('full_name')
        profile.address = request.form.get('address')
        profile.phone = request.form.get('phone')
        profile.bio = request.form.get('bio')
        db.session.commit()
        flash('Profile updated successfully!')
        return redirect(url_for('profile.manage_profile'))

    return render_template('profile.html', profile=profile)

@profile_bp.route('/portfolio/<int:portfolio_id>')
@login_required
def get_portfolio_item(portfolio_id):
    """Get single portfolio item for editing"""
    try:
        portfolio = Portfolio.query.filter_by(
            id=portfolio_id, 
            user_id=current_user.id
        ).first()
        
        if not portfolio:
            return jsonify({
                'success': False,
                'message': 'Portfolio item not found'
            }), 404
        
        return jsonify({
            'success': True,
            'portfolio': {
                'id': portfolio.id,
                'main_category': portfolio.main_category,
                'sub_category': portfolio.sub_category,
                'title': portfolio.title,
                'description': portfolio.description,
                'created_at': portfolio.created_at.isoformat()
            }
        })
        
    except Exception as e:
        print(f"❌ Error fetching portfolio item: {e}")
        return jsonify({
            'success': False,
            'message': 'Error fetching portfolio item'
        }), 500

@profile_bp.route('/portfolio/<int:portfolio_id>', methods=['PUT'])
@login_required
def update_portfolio_item(portfolio_id):
    """Update existing portfolio item"""
    try:
        portfolio = Portfolio.query.filter_by(
            id=portfolio_id,
            user_id=current_user.id
        ).first()
        
        if not portfolio:
            return jsonify({
                'success': False,
                'message': 'Portfolio item not found'
            }), 404
        
        # Get form data
        main_category = request.form.get('main_category')
        sub_category = request.form.get('sub_category')
        title = request.form.get('title')
        description = request.form.get('description', '')
        
        # Validation
        if not all([main_category, sub_category, title]):
            return jsonify({
                'success': False,
                'message': 'Main category, sub category, and title are required'
            }), 400
        
        # Update portfolio item
        portfolio.main_category = main_category
        portfolio.sub_category = sub_category
        portfolio.title = title
        portfolio.description = description
        
        db.session.commit()
        
        print(f"✅ Portfolio item updated: {title}")
        
        return jsonify({
            'success': True,
            'message': 'Portfolio item updated successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        print(f"❌ Error updating portfolio item: {e}")
        return jsonify({
            'success': False,
            'message': 'Error updating portfolio item'
        }), 500

@profile_bp.route('/portfolio/<int:portfolio_id>', methods=['DELETE'])
@login_required
def delete_portfolio_item(portfolio_id):
    """Delete portfolio item"""
    try:
        portfolio = Portfolio.query.filter_by(
            id=portfolio_id,
            user_id=current_user.id
        ).first()
        
        if not portfolio:
            return jsonify({
                'success': False,
                'message': 'Portfolio item not found'
            }), 404
        
        # Delete portfolio item
        db.session.delete(portfolio)
        db.session.commit()
        
        print(f"✅ Portfolio item deleted: {portfolio.title}")
        
        return jsonify({
            'success': True,
            'message': 'Portfolio item deleted successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        print(f"❌ Error deleting portfolio item: {e}")
        return jsonify({
            'success': False,
            'message': 'Error deleting portfolio item'
        }), 500