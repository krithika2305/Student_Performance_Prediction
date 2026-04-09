from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import traceback

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Load the trained model and required objects
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'model', 'scaler.pkl')
LABEL_ENCODER_PATH = os.path.join(os.path.dirname(__file__), 'model', 'label_encoder.pkl')
FEATURE_COLUMNS_PATH = os.path.join(os.path.dirname(__file__), 'model', 'feature_columns.pkl')

model = None
scaler = None
label_encoder = None
feature_columns = None

try:
    # Try to load model
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print("✓ Model loaded successfully")
    else:
        print("⚠ Model file not found at:", MODEL_PATH)
    
    # Try to load scaler
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print("✓ Scaler loaded successfully")
    else:
        print("⚠ Scaler file not found at:", SCALER_PATH)
    
    # Try to load label encoder
    if os.path.exists(LABEL_ENCODER_PATH):
        with open(LABEL_ENCODER_PATH, 'rb') as f:
            label_encoder = pickle.load(f)
        print("✓ Label encoder loaded successfully")
    else:
        print("⚠ Label encoder file not found at:", LABEL_ENCODER_PATH)
    
    # Try to load feature columns
    if os.path.exists(FEATURE_COLUMNS_PATH):
        with open(FEATURE_COLUMNS_PATH, 'rb') as f:
            feature_columns = pickle.load(f)
        print("✓ Feature columns loaded successfully")
        print("Features:", feature_columns)
    else:
        print("⚠ Feature columns file not found at:", FEATURE_COLUMNS_PATH)
        
except Exception as e:
    print(f"Error loading model files: {e}")
    traceback.print_exc()

# Helper mappings for training categories
def map_attendance(attendance):
    if attendance >= 95:
        return '95%+'
    if attendance >= 86:
        return '86-95%'
    if attendance >= 76:
        return '76-85%'
    if attendance >= 66:
        return '66-75%'
    if attendance >= 51:
        return '51-65%'
    return '0-50%'

def map_internal_marks(marks):
    if marks >= 25:
        return '25-30'
    if marks >= 16:
        return '16-25'
    if marks >= 6:
        return '6-15'
    return '0-5'

def map_study_hours(hours):
    if hours > 3:
        return 'More than 3 hours'
    if hours >= 2:
        return '2-3 hour'
    if hours >= 1:
        return '1 hour'
    return 'Less than one hour'

def map_sleep_hours(hours):
    if hours > 7:
        return 'More than 7 hours'
    if hours >= 6:
        return '6-7 hours'
    if hours >= 5:
        return '5-6 hours'
    return 'Less than 5 hours'

def map_mobile_usage(hours):
    if hours > 3:
        return 'More than 3 hours'
    if hours >= 2:
        return '2-3 hours'
    if hours >= 1:
        return '1-2 hours'
    return 'Less than 1 hours'

def map_travel_time(hours):
    if hours > 1:
        return 'more than 1 hour'
    if hours >= 0.5:
        return '30-60 minutes'
    if hours >= 0.25:
        return '15-30 minutes'
    return 'less than 15 minutes'

SEMESTER_MAP = {
    1: '1st',
    2: '2',
    3: '3rd',
    4: '4th',
    5: '5th',
    6: '6th',
    7: '7',
    8: '8'
}

def map_semester(semester):
    return SEMESTER_MAP.get(semester, str(semester))

def map_assignment_completion(value):
    normalized = value.strip().lower()
    if normalized in ['always', 'all']:
        return 'All'
    if normalized in ['mostly', 'most', 'sometimes']:
        return 'Most'
    if normalized in ['rarely', 'very few', 'few']:
        return 'Very few'
    return 'Most'


def map_doubt_method(value):
    normalized = value.strip().lower()
    if normalized in ['direct discussion', 'very frequently', 'very frequent', 'very often']:
        return 'Very frequently'
    if normalized in ['email', 'online platform', 'online']:
        return 'Frequently'
    if normalized in ['don\'t ask', 'dont ask', 'never']:
        return 'Never'
    return 'Sometimes'


def map_cgpa(cgpa):
    if feature_columns is None:
        return str(cgpa)
    cgpa_labels = [c.replace('CGPA_', '') for c in feature_columns if c.startswith('CGPA_')]
    numeric_labels = []
    for label in cgpa_labels:
        try:
            numeric_labels.append((float(label), label))
        except ValueError:
            continue
    if not numeric_labels:
        return str(cgpa)
    best = min(numeric_labels, key=lambda x: abs(x[0] - cgpa))[1]
    return best

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None or label_encoder is None or feature_columns is None:
        session['error'] = 'Model files not loaded. Please ensure all model, scaler, label_encoder, and feature_columns files exist.'
        return redirect(url_for('result'))
    
    try:
        # Get form data
        age = int(request.form.get('age', 20))
        gender = request.form.get('gender', 'Male')
        semester = int(request.form.get('semester', 1))
        attendance = int(request.form.get('attendance', 75))
        internal_marks = int(request.form.get('internal_marks', 15))
        assignment_completion = request.form.get('assignment_completion', 'Most')
        participation = request.form.get('participation', 'Sometimes')
        study_hours = int(request.form.get('study_hours', 3))
        doubt_method = request.form.get('doubt_method', 'Sometimes')
        teacher_help = request.form.get('teacher_help', 'Yes')
        sleep_hours = int(request.form.get('sleep_hours', 6))
        mobile_usage = int(request.form.get('mobile_usage', 2))
        travel_time = float(request.form.get('travel_time', 1))
        extracurricular = request.form.get('extracurricular', 'Sometimes')
        motivation_level = int(request.form.get('motivation_level', 3))
        stress_level = int(request.form.get('stress_level', 3))
        confidence_level = int(request.form.get('confidence_level', 3))
        cgpa = float(request.form.get('cgpa', 7.0))

        # Map numeric inputs into categorical training labels
        attendance_label = map_attendance(attendance)
        internal_marks_label = map_internal_marks(internal_marks)
        assignment_completion_label = map_assignment_completion(assignment_completion)
        participation_label = participation
        study_hours_label = map_study_hours(study_hours)
        doubt_method_label = map_doubt_method(doubt_method)
        teacher_help_label = teacher_help
        sleep_hours_label = map_sleep_hours(sleep_hours)
        mobile_usage_label = map_mobile_usage(mobile_usage)
        travel_time_label = map_travel_time(travel_time)
        extracurricular_label = extracurricular
        semester_label = map_semester(semester)
        cgpa_label = map_cgpa(cgpa)

        # Create DataFrame with the same structure as training data
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Semester': [semester_label],
            'Attendance': [attendance_label],
            'Internal_Marks': [internal_marks_label],
            'Assignment_Completion': [assignment_completion_label],
            'Participation': [participation_label],
            'Study_Hours': [study_hours_label],
            'Doubt_Method': [doubt_method_label],
            'Teacher_Help': [teacher_help_label],
            'Sleep_Hours': [sleep_hours_label],
            'Mobile_Usage': [mobile_usage_label],
            'Travel_Time': [travel_time_label],
            'Extracurricular': [extracurricular_label],
            'Motivation_Level': [motivation_level],
            'Stress_Level': [stress_level],
            'Confidence_Level': [confidence_level],
            'CGPA': [cgpa_label]
        })

        # One-hot encode categorical variables
        input_encoded = pd.get_dummies(input_data, drop_first=True)
        
        # Ensure all feature columns match training data
        # Add missing columns with 0 values and remove extra columns
        for col in feature_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        
        # Reorder columns to match training data
        input_encoded = input_encoded[feature_columns]
        
        # Convert to numpy array
        input_array = input_encoded.values
        
        # Scale the features
        input_scaled = scaler.transform(input_array)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        confidence = 'N/A'

        try:
            prediction_proba = model.predict_proba(input_scaled)[0]
            confidence = f"{max(prediction_proba) * 100:.2f}%"
        except Exception as proba_error:
            print(f"⚠ predict_proba failed: {proba_error}")
            confidence = 'N/A'

        # Convert model output to original label using the saved label encoder
        try:
            prediction_label = label_encoder.inverse_transform([int(prediction)])[0]
        except Exception as label_error:
            print(f"⚠ label encoder transform failed: {label_error}")
            prediction_label = str(prediction)

        # Store results in session
        session['prediction'] = prediction_label
        session['confidence'] = confidence
        session['student_data'] = {
            'age': age,
            'attendance': attendance,
            'internal_marks': internal_marks,
            'cgpa': cgpa,
            'study_hours': study_hours
        }
        session['error'] = None
        
        print(f"✓ Prediction successful: {prediction_label} (Confidence: {confidence})")
        
        return redirect(url_for('result'))
    
    except Exception as e:
        print(f"✗ Prediction error: {e}")
        traceback.print_exc()
        session['error'] = f"Prediction error: {str(e)}"
        return redirect(url_for('result'))

@app.route('/result')
def result():
    prediction = session.get('prediction', 'N/A')
    confidence = session.get('confidence', 'N/A')
    student_data = session.get('student_data', {})
    error = session.get('error', None)
    
    return render_template('result.html', 
                         prediction=prediction, 
                         confidence=confidence,
                         student_data=student_data,
                         error=error)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
