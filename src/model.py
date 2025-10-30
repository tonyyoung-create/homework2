"""
Model loading and prediction functionality.
"""
import os
import joblib
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

def load_model(model_path='models/spam_detector.joblib'):
    """
    Load the trained spam detection model.
    
    Args:
        model_path (str): Path to the saved model
        
    Returns:
        tuple: (model, vectorizer)
    """
    # If model file is missing, attempt to fetch from environment-provided URLs
    if not os.path.exists(model_path):
        model_url = os.environ.get('MODEL_URL')
        vec_url = os.environ.get('VECTORIZER_URL')
        if model_url and vec_url:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            try:
                # Download model
                r = requests.get(model_url, stream=True, timeout=30)
                r.raise_for_status()
                with open(model_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                # Download vectorizer
                vec_path = model_path.replace('.joblib', '_vectorizer.joblib')
                r = requests.get(vec_url, stream=True, timeout=30)
                r.raise_for_status()
                with open(vec_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            except Exception:
                # If download fails, continue and return (None, None) below
                pass

    if not os.path.exists(model_path):
        return None, None

    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(model_path.replace('.joblib', '_vectorizer.joblib'))
        return model, vectorizer
    except Exception:
        return None, None

def predict(text, model, vectorizer):
    """
    Make spam prediction on input text.
    
    Args:
        text (str): Preprocessed email content
        model: Trained model
        vectorizer: Fitted vectorizer
        
    Returns:
        tuple: (prediction, probability)
    """
    if model is None or vectorizer is None:
        # Model not available: return unknown prediction
        return 'unknown', [0.5, 0.5]

    # Transform text using vectorizer
    X = vectorizer.transform([text])

    # Make prediction
    prediction = model.predict(X)[0]
    try:
        probability = model.predict_proba(X)[0]
    except Exception:
        # Some classifiers don't implement predict_proba
        probability = None

    return prediction, probability

def train_model(X_train, y_train):
    """
    Train spam detection model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        tuple: (trained_model, fitted_vectorizer)
    """
    # Create and fit vectorizer
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_vec = vectorizer.fit_transform(X_train)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_vec, y_train)
    
    return model, vectorizer