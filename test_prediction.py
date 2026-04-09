"""
Test prediction script - validates that the model can make predictions correctly
Run this after check_model.py passes
"""

import pickle
import pandas as pd
import numpy as np
import os
import traceback

def load_model_components():
    """Load all saved model components"""
    try:
        with open('model/model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('model/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('model/label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
        with open('model/feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        return model, scaler, le, feature_columns
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None, None, None, None

def test_prediction():
    """Test a sample prediction"""
    
    print("=" * 60)
    print("PREDICTION TEST")
    print("=" * 60)
    
    model, scaler, le, feature_columns = load_model_components()
    
    if None in (model, scaler, le, feature_columns):
        print("✗ Cannot proceed - model components not loaded")
        return False
    
    try:
        # Create sample input data
        sample_input = pd.DataFrame({
            'Age': [20],
            'Gender': ['Male'],
            'Semester': [3],
            'Attendance': [80],
            'Internal_Marks': [16],
            'Assignment_Completion': ['Mostly'],
            'Participation': ['Often'],
            'Study_Hours': [4],
            'Doubt_Method': ['Direct Discussion'],
            'Teacher_Help': ['Yes'],
            'Sleep_Hours': [7],
            'Mobile_Usage': [3],
            'Travel_Time': [1.0],
            'Extracurricular': ['Sometimes'],
            'Motivation_Level': ['High'],
            'Stress_Level': ['Low'],
            'Confidence_Level': ['High'],
            'CGPA': [8.5]
        })
        
        print("\n📋 Sample Input:")
        print(sample_input.to_string())
        
        # One-hot encode
        sample_encoded = pd.get_dummies(sample_input, drop_first=True)
        
        # Match feature columns
        for col in feature_columns:
            if col not in sample_encoded.columns:
                sample_encoded[col] = 0
        
        sample_encoded = sample_encoded[feature_columns]
        
        # Scale
        sample_scaled = scaler.transform(sample_encoded.values)
        
        # Predict
        prediction = model.predict(sample_scaled)[0]
        probabilities = model.predict_proba(sample_scaled)[0]
        
        # Map to label
        pred_label = le.inverse_transform([int(prediction)])[0]
        
        print("\n✓ PREDICTION SUCCESSFUL!")
        print(f"\nPredicted Performance Level: {pred_label}")
        print(f"Prediction Probability: {max(probabilities) * 100:.2f}%")
        
        print("\nAll Class Probabilities:")
        for i, prob in enumerate(probabilities):
            label = le.inverse_transform([i])[0]
            print(f"  {label}: {prob * 100:.2f}%")
        
        print("\n" + "=" * 60)
        print("✓ Model is working correctly!")
        print("You can now run: python app.py")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ Prediction failed!")
        print(f"Error: {e}")
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_prediction()
