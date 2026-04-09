"""
This script saves the trained model and all necessary objects for prediction.
Run this after training your model in the notebook.
"""

import pickle
import os
from pathlib import Path

# Create model directory if it doesn't exist
model_dir = Path('model')
model_dir.mkdir(exist_ok=True)

# You should run this in your notebook after training
# Replace the variables with your actual trained objects

def save_model_objects(model, scaler, label_encoder, X_train):
    """
    Save all necessary model objects
    
    Parameters:
    - model: trained sklearn model
    - scaler: StandardScaler object
    - label_encoder: LabelEncoder object for Performance_Level
    - X_train: training features DataFrame (after one-hot encoding)
    """
    
    try:
        # Save model
        with open('model/model.pkl', 'wb') as f:
            pickle.dump(model, f)
        print("✓ Model saved")
        
        # Save scaler
        with open('model/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        print("✓ Scaler saved")
        
        # Save label encoder
        with open('model/label_encoder.pkl', 'wb') as f:
            pickle.dump(label_encoder, f)
        print("✓ Label encoder saved")
        
        # Save feature columns (list of feature names after one-hot encoding)
        feature_columns = list(X_train.columns)
        with open('model/feature_columns.pkl', 'wb') as f:
            pickle.dump(feature_columns, f)
        print("✓ Feature columns saved")
        print(f"\nFeature columns ({len(feature_columns)}): {feature_columns}")
        
        print("\n✓ All model files saved successfully to 'model/' directory!")
        
    except Exception as e:
        print(f"✗ Error saving model: {e}")
        raise

# Example usage (uncomment after training):
# save_model_objects(model, scaler, le, X)

if __name__ == '__main__':
    print("Import this script in your notebook and call save_model_objects()")
    print("after training your model.")
    print("\nExample:")
    print("  from save_model import save_model_objects")
    print("  save_model_objects(model, scaler, le, X_train)")
