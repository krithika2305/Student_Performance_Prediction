"""
Diagnostic script to check if all model files are properly loaded
Run this before starting the Flask app
"""

import pickle
import os
import sys

def check_model_files():
    print("=" * 60)
    print("MODEL FILES DIAGNOSTIC")
    print("=" * 60)
    
    model_dir = 'model'
    files_to_check = {
        'model.pkl': 'Trained Model (LogisticRegression)',
        'scaler.pkl': 'StandardScaler Object',
        'label_encoder.pkl': 'LabelEncoder Object',
        'feature_columns.pkl': 'Feature Column Names'
    }
    
    all_good = True
    
    for filename, description in files_to_check.items():
        filepath = os.path.join(model_dir, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    obj = pickle.load(f)
                size = os.path.getsize(filepath)
                print(f"✓ {filename:<30} ({size:,} bytes)")
                print(f"  → {description}")
                
                if filename == 'feature_columns.pkl':
                    print(f"  → Columns: {len(obj)} features")
                    
            except Exception as e:
                print(f"✗ {filename:<30} - ERROR LOADING")
                print(f"  → {str(e)}")
                all_good = False
        else:
            print(f"✗ {filename:<30} - NOT FOUND")
            print(f"  → Expected at: {filepath}")
            all_good = False
    
    print("\n" + "=" * 60)
    
    if all_good:
        print("✓ ALL MODEL FILES FOUND AND VALID!")
        print("\nYou can now start the Flask app:")
        print("  python app.py")
    else:
        print("✗ MISSING OR INVALID MODEL FILES")
        print("\nFix:")
        print("1. Run your notebook: notebooks/Student_Performance_Prediction.ipynb")
        print("2. Execute all cells including the model training cells")
        print("3. Run the final cell to save the model files")
        print("4. Then try this diagnostic again")
    
    print("=" * 60)
    return all_good

if __name__ == '__main__':
    success = check_model_files()
    sys.exit(0 if success else 1)
