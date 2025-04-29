# app.py
import matplotlib
matplotlib.use('Agg') # Use non-GUI backend
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import csv
from io import StringIO
from flask import make_response
from sqlalchemy import func  
import os
import tensorflow as tf
import torch
import torch.nn.functional as F
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate 
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from torchvision import transforms
from PIL import Image
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
import re
from flask import current_app as app
import logging
from sqlalchemy.sql import text
from flask_login import UserMixin

# Flask app configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this in production
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:root@localhost/alzheimer_db'  # MySQL config
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'model/ResNet_Model_92_.keras'
IMG_SIZE = (224, 224)
BRAIN_MRI_FOLDER = r'C:\Users\Administrator\Desktop\FINAL YEAR PROJECT\Deploy\static\brain_mri_referance'  # Reference folder for brain MRI images
SIMILARITY_THRESHOLD = 0.70  # Minimum similarity score to consider as valid brain MRI

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(BRAIN_MRI_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize extensions
db = SQLAlchemy(app)
migrate = Migrate(app, db)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Global variables
model = None
CLASS_NAMES = ["Mild Impairment", "Moderate Impairment", "No Impairment", "Very Mild Impairment"]

# Image transformations for similarity check
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor()
])

# Database Models
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(256), nullable=False)

    def set_password(self, password):
        self.password = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password, password)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_name = db.Column(db.String(100), nullable=False)
    image_path = db.Column(db.String(500), nullable=False)
    prediction_result = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    visualization_path = db.Column(db.String(500))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

class PatientHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, nullable=False)
    notes = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# User loader for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Model loading function
def load_model_safe():
    global model
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("Model loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Image processing functions for the existing pipeline
def preprocess_image(image_path):
    try:
        image = load_img(image_path, target_size=IMG_SIZE)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = image / 255.0
        return image
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# Preprocess image for similarity comparison
def preprocess_image_for_similarity(image_path):
    try:
        # Verify if the file is a valid image before opening
        with Image.open(image_path) as img:
            img.verify()  # Verify image integrity

        # Re-open the image after verification
        image = Image.open(image_path).convert("L")
        image = transform(image).to(device)
        return image
    except (IOError, OSError, Image.UnidentifiedImageError) as e:
        print(f"Warning: Skipping invalid image: {image_path} ({e})")
        return None

# Compute cosine similarity between two images
def compute_similarity(img1, img2):
    img1_flat = img1.view(1, -1)  # Flatten images
    img2_flat = img2.view(1, -1)
    similarity = F.cosine_similarity(img1_flat, img2_flat).item()
    return similarity

# Build reference feature set from brain MRI images
def build_reference_features():
    ref_features = []
    for filename in os.listdir(BRAIN_MRI_FOLDER):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(BRAIN_MRI_FOLDER, filename)
            feature = preprocess_image_for_similarity(img_path)
            if feature is not None:
                ref_features.append(feature)
    return ref_features

# Check if an image is a brain MRI based on similarity
def is_brain_mri(image_path):
    # Load reference features
    ref_features = build_reference_features()
    if not ref_features:
        print("Warning: No reference brain MRI images found. Skipping similarity check.")
        return True  # Skip check if no reference images
    
    # Load test image
    test_image = preprocess_image_for_similarity(image_path)
    if test_image is None:
        return False
    
    # Compute similarities
    similarities = []
    for ref_img in ref_features:
        similarity = compute_similarity(test_image, ref_img)
        similarities.append(similarity)
    
    # Calculate average similarity
    avg_similarity = np.mean(similarities)
    print(f"Image: {os.path.basename(image_path)}, Similarity: {avg_similarity:.2f}")
    
    return avg_similarity >= SIMILARITY_THRESHOLD

def visualize_prediction(image_path, model):
    try:
        image_array = preprocess_image(image_path)
        predictions = model.predict(image_array)
        confidence_scores = predictions[0] * 100
        predicted_class_idx = np.argmax(predictions[0])

        original_image = load_img(image_path, target_size=IMG_SIZE)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title("MRI Image", fontsize=14)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        bar_positions = np.arange(len(CLASS_NAMES))
        plt.bar(bar_positions, confidence_scores, color='skyblue', alpha=0.8)
        plt.xticks(bar_positions, CLASS_NAMES, rotation=45, fontsize=12)
        plt.title(f"Predicted: {CLASS_NAMES[predicted_class_idx]}\nConfidence: {confidence_scores[predicted_class_idx]:.2f}%", fontsize=14)
        plt.ylabel("Confidence (%)", fontsize=12)
        plt.ylim(0, 100)

        input_filename = os.path.splitext(os.path.basename(image_path))[0]
        visualization_path = os.path.join(UPLOAD_FOLDER, f"viz_{input_filename}.png")
        
        plt.tight_layout()
        plt.savefig(visualization_path)
        plt.close()
        return visualization_path
    except Exception as e:
        print(f"Error generating visualization: {e}")
        return None

# Routes
@app.route('/')
def home():
    if not model:
        return render_template('index.html', error="Model not loaded. Please contact administrator.")
    return render_template('index.html')

# Register Route
# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            email = request.form.get('email', '').strip()
            name = request.form.get('name', '').strip()
            password = request.form.get('password')
            confirm_password = request.form.get('confirm_password')
            
            # Generate username from email
            username = email.split('@')[0] if email else None
            
            # Validation checks
            if not all([email, name, password, confirm_password]):
                flash('All fields are required.', 'danger')
                return render_template('register.html')
            
            if password != confirm_password:
                flash('Passwords do not match.', 'danger')
                return render_template('register.html')
            
            # Check for existing user
            existing_user = User.query.filter(
                (User.email == email) | (User.username == username)
            ).first()
            
            if existing_user:
                flash('Email or username already registered.', 'danger')
                return render_template('register.html')
            
            # Create new user
            user = User(
                username=username,
                name=name,
                email=email
            )
            user.set_password(password)
            
            db.session.add(user)
            db.session.commit()
            
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
            
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Registration error: {str(e)}")
            flash('An error occurred during registration. Please try again.', 'danger')
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password', 'danger')
    return render_template('login.html')

@app.route('/verify_logout', methods=['GET', 'POST'])
@login_required
def verify_logout():
    if request.method == 'POST':
        password = request.form.get('password')

        # Verify password for current logged-in user
        if current_user.check_password(password):
            # Clear all session data and log out
            session.clear()
            logout_user()
            
            return redirect(url_for('home'))
        else:
            flash('Incorrect password, please try again.', 'danger')

    return render_template('verify_logout.html')

@app.route('/logout')
@login_required
def logout():
    return redirect(url_for('verify_logout'))

@app.route('/verify_dashboard', methods=['GET', 'POST'])
@login_required
def verify_dashboard():
    # Store the current timestamp when verification is requested
    session['verification_requested'] = datetime.now().timestamp()
    
    if request.method == 'POST':
        password = request.form.get('password')

        # Verify password for current logged-in user
        if current_user.check_password(password):
            # Set verification timestamp
            session['last_verified'] = datetime.now().timestamp()
            return redirect(url_for('dashboard'))
        else:
            flash('Incorrect password, please try again.', 'danger')

    return render_template('verify_dashboard.html')

@app.route('/dashboard') 
@login_required
def dashboard():
    # Check verification (keep existing verification logic)
    last_verified = session.get('last_verified')
    verification_requested = session.get('verification_requested')

    if (last_verified is None or 
        verification_requested is None or 
        last_verified < verification_requested):
        return redirect(url_for('verify_dashboard'))

    # Clear verification after showing dashboard
    session.pop('last_verified', None)
    session.pop('verification_requested', None)

    # Get the current user's ID
    current_user_id = current_user.id

    # Fetch latest prediction for each patient (based on image filename)
    # Group predictions by patient name and select the latest result for each patient
    subquery = db.session.query(
        Prediction.patient_name,
        func.max(Prediction.timestamp).label('latest')
    ).filter(Prediction.doctor_id == current_user_id).group_by(Prediction.patient_name).subquery()

    predictions = db.session.query(Prediction).join(
        subquery, 
        (Prediction.patient_name == subquery.c.patient_name) & 
        (Prediction.timestamp == subquery.c.latest)
    ).order_by(Prediction.id.desc()).all()

    # Extract patient name from image file name
    for prediction in predictions:
        prediction.patient_name = prediction.image_path.split('/')[-1].split('.')[0]  # Extract filename

    # Get trend data
    trend_data = db.session.query(
        func.date(Prediction.timestamp),
        func.count(Prediction.id)
    ).filter(Prediction.doctor_id == current_user_id).group_by(
        func.date(Prediction.timestamp)
    ).all()

    trend_labels = [date.strftime('%Y-%m-%d') for date, _ in trend_data]
    trend_counts = [count for _, count in trend_data]

    # Filter distribution data by current user
    distribution_data = {
        'No Impairment': Prediction.query.filter_by(doctor_id=current_user_id, prediction_result='No Impairment').count(),
        'Very Mild Impairment': Prediction.query.filter_by(doctor_id=current_user_id, prediction_result='Very Mild Impairment').count(),
        'Mild Impairment': Prediction.query.filter_by(doctor_id=current_user_id, prediction_result='Mild Impairment').count(),
        'Moderate Impairment': Prediction.query.filter_by(doctor_id=current_user_id, prediction_result='Moderate Impairment').count()
    }

    # Get most common prediction
    most_common = db.session.query(
        Prediction.prediction_result,
        func.count(Prediction.prediction_result).label('count')
    ).filter(Prediction.doctor_id == current_user_id).group_by(Prediction.prediction_result).order_by(func.count(Prediction.prediction_result).desc()).first()

    most_common_prediction = most_common[0] if most_common else "N/A"

    return render_template('dashboard.html',
                    predictions=predictions,
                    trend_labels=trend_labels,
                    trend_data=trend_counts,
                    distribution_data=list(distribution_data.values()),
                    distribution_labels=list(distribution_data.keys()),
                    most_common_prediction=most_common_prediction)

@app.route('/delete_prediction/<int:prediction_id>', methods=['POST'])
@login_required
def delete_prediction(prediction_id):
    try:
        # Retrieve prediction by ID
        prediction = Prediction.query.get(prediction_id)
        
        if not prediction:
            return jsonify({"success": False, "message": "Prediction not found"}), 404

        # Delete the prediction entry only
        db.session.delete(prediction)
        db.session.commit()

        return jsonify({"success": True, "message": "Prediction deleted successfully"})

    except Exception as e:
        db.session.rollback()
        print(f"Error deleting prediction: {e}")
        return jsonify({"success": False, "message": f"Error deleting prediction: {str(e)}"}), 500


@app.route('/update_settings', methods=['POST'])
@login_required
def update_settings():
    user = current_user
    name = request.form.get('name')
    username = request.form.get('username')
    new_password = request.form.get('new_password')
    confirm_password = request.form.get('confirm_password')

    try:
        if name:
            user.name = name

        if username and username != user.username:
            existing_user = User.query.filter_by(username=username).first()
            if existing_user and existing_user != user:
                flash('Username already taken', 'danger')
                return redirect(url_for('dashboard'))
            user.username = username

        if new_password:
            if new_password != confirm_password:
                flash('Passwords do not match', 'danger')
                return redirect(url_for('dashboard'))
            user.set_password(new_password)  # Using the new set_password method

        db.session.commit()
        flash('Settings updated successfully', 'success')
    except Exception as e:
        db.session.rollback()
        flash('An error occurred while updating settings', 'danger')
        print(f"Error updating settings: {str(e)}")

    return redirect(url_for('dashboard'))

@app.route('/export-predictions')
@login_required
def export_predictions():
    # Create a string buffer to write CSV data
    si = StringIO()
    writer = csv.writer(si)
    
    # Write headers
    writer.writerow(['ID', 'Date', 'Patient', 'Prediction Result', 'Confidence'])
    
    # Write prediction data
    predictions = Prediction.query.order_by(Prediction.id.desc()).all()  # Added descending order
    for prediction in predictions:
        # Extract patient name from image path - remove extension
        patient_name = os.path.splitext(os.path.basename(prediction.image_path))[0]
        
        writer.writerow([
            prediction.id,
            prediction.timestamp.strftime('%Y-%m-%d'),
            patient_name,  # Using the extracted patient name
            prediction.prediction_result,
            f"{prediction.confidence:.1f}%"
        ])
    
    # Create the response
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=predictions.csv"
    output.headers["Content-type"] = "text/csv"
    
    return output


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if not model:
        return render_template('index.html', error="Model not loaded. Please contact administrator.")

    try:
        patient_name = request.form.get('patient_name', 'Unknown')
        
        # Handle single file upload
        single_file = request.files.get('file')
        if single_file and single_file.filename != '':
            filename = secure_filename(single_file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            single_file.save(file_path)

            # Check if the image is a brain MRI using similarity verification
            valid_mri = is_brain_mri(file_path)
            
            if not valid_mri:
                return render_template('index.html', 
                                      error=f"The uploaded image '{filename}' does not appear to be a brain MRI. Please upload a valid brain MRI image.")

            image = preprocess_image(file_path)
            if image is not None:
                predictions = model.predict(image)
                predicted_class_idx = np.argmax(predictions[0])
                stage_name = CLASS_NAMES[predicted_class_idx]
                confidence = float(np.max(predictions[0]) * 100)

                visualization_path = visualize_prediction(file_path, model)
                
                # Save prediction to database
                prediction = Prediction(
                    patient_name=patient_name,
                    image_path=filename,
                    prediction_result=stage_name,
                    confidence=confidence,
                    visualization_path=os.path.basename(visualization_path) if visualization_path else None,
                    doctor_id=current_user.id
                )
                db.session.add(prediction)
                db.session.commit()

                return render_template(
                    'result.html',
                    image_path=url_for('static', filename=f'uploads/{filename}'),
                    prediction=stage_name,
                    confidence=confidence,
                    visualization_path=url_for('static', filename=f'uploads/{os.path.basename(visualization_path)}') if visualization_path else None
                )

        # Handle folder upload
        files = request.files.getlist('folder[]')
        if files and len(files) > 0 and any(f.filename != '' for f in files):
            predictions_results = []
            skipped_files = []
            
            for file in files:
                if file and file.filename != '':
                    try:
                        filename = secure_filename(file.filename)
                        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        file.save(file_path)

                        # Check if the image is a brain MRI using similarity verification
                        valid_mri = is_brain_mri(file_path)
                        
                        if not valid_mri:
                            skipped_files.append(f"{filename} (Not a valid brain MRI)")
                            continue

                        image = preprocess_image(file_path)
                        if image is not None:
                            predictions = model.predict(image)
                            predicted_class_idx = np.argmax(predictions[0])
                            stage_name = CLASS_NAMES[predicted_class_idx]
                            confidence = float(np.max(predictions[0]) * 100)
                            
                            visualization_path = visualize_prediction(file_path, model)
                            
                            # Save prediction to database
                            prediction = Prediction(
                                patient_name=f"{patient_name}_{filename}",
                                image_path=filename,
                                prediction_result=stage_name,
                                confidence=confidence,
                                visualization_path=os.path.basename(visualization_path) if visualization_path else None,
                                doctor_id=current_user.id
                            )
                            db.session.add(prediction)
                            
                            predictions_results.append({
                                'filename': filename,
                                'image_path': url_for('static', filename=f'uploads/{filename}'),
                                'prediction': stage_name,
                                'confidence': confidence,
                                'visualization_path': url_for('static', filename=f'uploads/{os.path.basename(visualization_path)}') if visualization_path else None
                            })
                    except Exception as e:
                        print(f"Error processing file {filename}: {str(e)}")
                        skipped_files.append(f"{filename} (Error: {str(e)})")
                        continue
                        
            db.session.commit()
            
            if predictions_results:
                return render_template('bulk_result.html', results=predictions_results, skipped_files=skipped_files)
            else:
                if skipped_files:
                    return render_template('index.html', 
                                         error=f"No valid brain MRI images were found in the upload. Skipped files: {', '.join(skipped_files)}")
                else:
                    return render_template('index.html', 
                                         error="No valid images were found in the uploaded folder.")

        return render_template('index.html', error="Please select a file or folder to upload.")

    except Exception as e:
        print(f"Upload error: {str(e)}")
        return render_template('index.html', error=f"Error processing upload: {str(e)}")


def init_db():
    # Create tables only if they don't exist
    db.create_all()

# Create a test user
def create_test_user(app):
    with app.app_context():
        try:
            # Check if test user already exists
            existing_user = User.query.filter_by(email='doctor@example.com').first()
            
            if not existing_user:
                # Create new test user with all required fields
                test_user = User(
                    username='doctor_smith',
                    name='Dr. Smith',
                    email='doctor@example.com'
                )
                test_user.set_password('password123')
                
                # Add and commit with error handling
                db.session.add(test_user)
                db.session.commit()
                print("Test user created successfully")
                
        except IntegrityError as e:
            db.session.rollback()
            print(f"Error creating test user: {str(e)}")
            
        except Exception as e:
            db.session.rollback()
            print(f"Unexpected error creating test user: {str(e)}")


if __name__ == '__main__':
    with app.app_context():
        init_db()  # Initialize database tables
        create_test_user(app)  # Create test user if it doesn't exist
    if load_model_safe():
        app.run(host='0.0.0.0', port=5000, debug=True)