from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import numpy as np
import joblib
import librosa
import pandas as pd
from sklearn.impute import SimpleImputer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load the pre-trained model from the notebook
model = joblib.load('parkinsons_detector.pkl')

# Function to extract voice features
def extract_voice_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    features = {
        'MDVP:Fo(Hz)': np.mean(librosa.feature.mfcc(y=y, sr=sr)),
        'MDVP:Fhi(Hz)': np.max(librosa.feature.mfcc(y=y, sr=sr)),
        'MDVP:Flo(Hz)': np.min(librosa.feature.mfcc(y=y, sr=sr)),
        'MDVP:Jitter(%)': np.mean(librosa.feature.mfcc(y=y, sr=sr)),
        'MDVP:Jitter(Abs)': np.mean(librosa.feature.mfcc(y=y, sr=sr)),
        'MDVP:RAP': np.mean(librosa.feature.mfcc(y=y, sr=sr)),
        'MDVP:PPQ': np.mean(librosa.feature.mfcc(y=y, sr=sr)),
        'Jitter:DDP': np.mean(librosa.feature.mfcc(y=y, sr=sr)),
        'MDVP:Shimmer': np.mean(librosa.feature.mfcc(y=y, sr=sr)),
        'MDVP:Shimmer(dB)': np.mean(librosa.feature.mfcc(y=y, sr=sr)),
        'Shimmer:APQ3': np.mean(librosa.feature.mfcc(y=y, sr=sr)),
        'Shimmer:APQ5': np.mean(librosa.feature.mfcc(y=y, sr=sr)),
        'MDVP:APQ': np.mean(librosa.feature.mfcc(y=y, sr=sr)),
        'Shimmer:DDA': np.mean(librosa.feature.mfcc(y=y, sr=sr)),
        'NHR': np.mean(librosa.feature.mfcc(y=y, sr=sr)),
        'HNR': np.mean(librosa.feature.mfcc(y=y, sr=sr)),
        'RPDE': np.mean(librosa.feature.mfcc(y=y, sr=sr)),
        'DFA': np.mean(librosa.feature.mfcc(y=y, sr=sr)),
        'spread1': np.mean(librosa.feature.mfcc(y=y, sr=sr)),
        'spread2': np.mean(librosa.feature.mfcc(y=y, sr=sr)),
        'D2': np.mean(librosa.feature.mfcc(y=y, sr=sr)),
        'PPE': np.mean(librosa.feature.mfcc(y=y, sr=sr))
    }
    return features

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle audio file upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    # Extract features from the audio file
    features = extract_voice_features(file_path)
    
    # Convert features to DataFrame
    df_features = pd.DataFrame([features])
    
    # Handle NaN values with imputation
    imputer = SimpleImputer(strategy='mean')
    df_features_imputed = pd.DataFrame(imputer.fit_transform(df_features))
    
    # Reshape features to match model input
    features_for_model = np.array(df_features_imputed.values).reshape(1, -1)
    
    # Handle edge case: check if there are still NaN values
    if np.isnan(features_for_model).any():
        return jsonify({'error': 'NaN values found in features'}), 400
    
    # Predict using the loaded model
    prediction = model.predict(features_for_model)
    
    # Remove the uploaded file after processing
    os.remove(file_path)
    
    # Prepare prediction result for rendering in a new template
    if prediction[0] == 1:
        prediction_text = "Parkinson's Disease"
    elif prediction[0] == 0 :
        prediction_text = "Healthy"
    else:
        prediction_text = "Unknown"
    
    # Render a new template with the prediction result
    return render_template('prediction_result.html', prediction=prediction_text)

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
