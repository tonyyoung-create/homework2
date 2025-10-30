"""
Model loading and prediction functionality.
"""
import os
import joblib
import numpy as np
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