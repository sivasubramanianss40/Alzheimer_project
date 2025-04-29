from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.sql import func

# Initialize the SQLAlchemy database

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    role = db.Column(db.String(20), default='doctor')  # For role-based access
    created_at = db.Column(db.DateTime, default=func.now())
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    predictions = db.relationship('Prediction', backref='doctor', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def update_last_login(self):
        self.last_login = datetime.utcnow()
        db.session.commit()

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_name = db.Column(db.String(100), nullable=False)
    patient_id = db.Column(db.String(50), nullable=False)  # Ensure this matches database.py
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    image_path = db.Column(db.String(500), nullable=False)
    prediction_result = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    visualization_path = db.Column(db.String(500))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)


    def to_dict(self):
        return {
            'id': self.id,
            'patient_name': self.patient_name,
            'patient_id': self.patient_id,
            'age': self.age,
            'gender': self.gender,
            'prediction_result': self.prediction_result,
            'confidence': self.confidence,
            'image_path': self.image_path,
            'visualization_path': self.visualization_path,
            'notes': self.notes,
            'timestamp': self.timestamp.isoformat(),
            'doctor_id': self.doctor_id
        }

class PatientHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(50), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    prediction_id = db.Column(db.Integer, db.ForeignKey('prediction.id'), nullable=False)
    visit_date = db.Column(db.DateTime, default=func.now())
    follow_up_notes = db.Column(db.Text)

    def to_dict(self):
        return {
            'id': self.id,
            'patient_id': self.patient_id,
            'doctor_id': self.doctor_id,
            'prediction_id': self.prediction_id,
            'visit_date': self.visit_date.isoformat(),
            'follow_up_notes': self.follow_up_notes
        }
