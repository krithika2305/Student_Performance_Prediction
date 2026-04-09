# Student Performance Prediction Web Application

A machine learning-based web application that predicts student academic performance based on various factors including attendance, study habits, mental health, and lifestyle factors.

## Features

- **Predictive Analysis**: Uses a trained Logistic Regression model to predict student performance levels (Poor, Average, Good, Excellent)
- **User-Friendly Interface**: Interactive web form to input student data
- **Personalized Recommendations**: Provides tailored suggestions based on predicted performance level
- **Real-time Predictions**: Instant prediction results with confidence scores

## Project Structure

```
ML_Project/
├── app.py                              # Flask application
├── requirements.txt                    # Python dependencies
├── README.md                          # Project documentation
├── .gitignore                         # Git ignore file
├── data/                              # Input datasets
├── model/                             # Trained model files
│   ├── student_performance_logistic_model.pkl
│   └── scaler.pkl
├── notebooks/                         # Jupyter notebooks
│   └── Student_Performance_Prediction.ipynb
├── static/                            # Static files
│   └── style.css                      # Styling
└── templates/                         # HTML templates
    ├── index.html                     # Prediction form
    └── result.html                    # Results page
```

## Input Features

The application uses 18 features to make predictions:

1. **Personal Information**
   - Age
   - Gender

2. **Academic Information**
   - Current Semester
   - Attendance (%)
   - Internal Marks (out of 20)
   - CGPA (out of 10)

3. **Learning Habits**
   - Study Hours Per Day
   - Assignment Completion Level
   - Lab/Seminar Participation
   - How to Ask Doubts

4. **Support System**
   - Help from Teachers/Classmates
   - Sleep Hours Per Day
   - Mobile/Social Media Usage
   - Travel Time

5. **Mental & Emotional Factors**
   - Motivation Level
   - Stress Level
   - Confidence Level
   - Extracurricular Activities

## Model Information

- **Algorithm**: Logistic Regression (Multi-class)
- **Training Data**: Student survey responses with 180+ samples
- **Target Variable**: Performance Level (4 classes)
- **Preprocessing**: 
  - Label Encoding for target variable
  - One-Hot Encoding for categorical features
  - StandardScaler for feature normalization

## Installation

1. **Clone or download the repository**
   ```bash
   cd ML_Project
   ```

2. **Create a virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start (Step-by-Step)

### Step 1: Train Your Model
1. Open the Jupyter notebook: `notebooks/Student_Performance_Prediction.ipynb`
2. **Run all cells** (click "Run All")
3. Wait for training to complete
4. The last cell saves all model files automatically

### Step 2: Verify Model Files
Before starting the Flask app, verify all model files are saved:

```bash
python check_model.py
```

You should see:
```
✓ model.pkl (2.5 MB)
✓ scaler.pkl (15 KB)
✓ label_encoder.pkl (1 KB)
✓ feature_columns.pkl (5 KB)
```

If you see ✗ (crosses), you need to run the notebook again and save the model files.

### Step 3: Run the Flask Application
```bash
python app.py
```

You should see:
```
✓ Model loaded successfully
✓ Scaler loaded successfully
✓ Label encoder loaded successfully
✓ Feature columns loaded successfully
Features: [... list of columns ...]
 * Running on http://localhost:5000/
```

### Step 4: Open in Browser
Navigate to: `http://localhost:5000`

Fill in the form and click "Predict Performance" to see results!

## Troubleshooting

### Problem: "Model files not loaded"
**Solution:**
1. Run your notebook: `notebooks/Student_Performance_Prediction.ipynb`
2. Execute all cells completely
3. Check that these files exist in `model/` folder:
   - `model.pkl`
   - `scaler.pkl`
   - `label_encoder.pkl`
   - `feature_columns.pkl`
4. Run diagnostic: `python check_model.py`

### Problem: "Port already in use"
**Solution:** Edit `app.py` line 120:
```python
app.run(debug=True, port=5001)  # Change 5000 to 5001 or any free port
```

### Problem: "ModuleNotFoundError: No module named 'flask'"
**Solution:** Install dependencies:
```bash
pip install -r requirements.txt
```

### Problem: Prediction returns error
**Solution:**
1. Check console for error messages
2. Verify all model files exist: `python check_model.py`
3. Restart Flask app: `python app.py`
4. Check your form input values

### Problem: Form not submitting
**Solution:**
1. Check browser console (F12 → Console tab)
2. Make sure all form fields are filled
3. Try in a different browser
4. Clear browser cache (Ctrl+Shift+Delete)

## File Structure Explained

## File Structure Explained

```
ML_Project/
├── app.py                              # Flask app (main entry point)
├── check_model.py                      # Diagnostic tool ⭐
├── save_model.py                       # Helper to save models
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── .gitignore                          # Git ignore patterns
│
├── data/                               # Your datasets
│
├── model/                              # Trained model files (generated by notebook)
│   ├── model.pkl                       # Trained Logistic Regression model
│   ├── scaler.pkl                      # StandardScaler for feature normalization
│   ├── label_encoder.pkl               # LabelEncoder for target variable
│   └── feature_columns.pkl             # Feature column names (important!)
│
├── notebooks/                          # Jupyter notebooks
│   └── Student_Performance_Prediction.ipynb  # Main training notebook ⭐
│
├── static/                             # Static files (CSS, JS, images)
│   └── style.css                       # Application styling
│
└── templates/                          # HTML templates
    ├── index.html                      # Prediction form
    └── result.html                     # Results and recommendations page
```

### Important Helper Scripts

These scripts help you debug and set up the application:

| Script | Purpose | When to Run |
|--------|---------|------------|
| **check_model.py** | ✓ Verify all model files are saved correctly | Before `python app.py` |
| **test_prediction.py** | ✓ Test that predictions work with sample data | After `check_model.py` passes |
| **save_model.py** | Reference for saving model components | Import in your notebook |

- **Excellent**: Score 3 - Outstanding academic performance (>90% predicted)
- **Good**: Score 2 - Strong academic performance (70-90% predicted)
- **Average**: Score 1 - Fair academic performance (50-70% predicted)
- **Poor**: Score 0 - Needs improvement (<50% predicted)

## Recommendations

The application provides specific recommendations based on predicted performance:

- **For Poor Performance**: Focus on attendance, study hours, and seeking help
- **For Average Performance**: Work on assignments, participation, and confidence
- **For Good Performance**: Explore advanced topics and maintain consistency
- **For Excellent Performance**: Help peers and explore research opportunities

## Technologies Used

- **Backend**: Flask (Python web framework)
- **Machine Learning**: Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Frontend**: HTML5, CSS3
- **Sessions**: Flask session management

## Dependencies

- Flask==3.0.0
- numpy==1.24.0
- pandas==2.0.0
- scikit-learn==1.3.0

## Future Enhancements

- [ ] Add more machine learning algorithms (Random Forest, XGBoost)
- [ ] Implement data visualization (performance distributions, charts)
- [ ] Add user authentication and profile management
- [ ] Store predictions in a database
- [ ] Mobile app version
- [ ] Real-time prediction accuracy metrics
- [ ] Feature importance analysis

## How the Model Works

1. **Data Collection**: Collects 18 features from student responses
2. **Preprocessing**: Applies encoding and scaling to standardize data
3. **Prediction**: Uses trained Logistic Regression model to predict performance
4. **Confidence Score**: Shows probability of the predicted class
5. **Recommendations**: Generates tailored advice based on prediction and input data

## Troubleshooting

### Model files not found
- Ensure `student_performance_logistic_model.pkl` and `scaler.pkl` are in the `model/` directory

### Port already in use
- Change the port in `app.py`: `app.run(debug=True, port=5001)`

### Missing dependencies
- Install all requirements: `pip install -r requirements.txt`

## Contributing

Feel free to fork, modify, and improve this project!


## Author

Created as part of ML Project - Student Performance Prediction Initiative
