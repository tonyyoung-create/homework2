"""
Main Streamlit application for spam email detection.
"""
import streamlit as st
import pandas as pd
import numpy as np
from preprocessing import preprocess_email, extract_features
from model import load_model, predict
from visualization import (
    plot_wordcloud,
    plot_feature_importance,
    plot_confusion_matrix,
    plot_roc_curve
)

st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📧",
    layout="wide"
)

def main():
    st.title("📧 Spam Email Detection System")
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Home", "Single Prediction", "Batch Analysis", "Model Performance"]
    )

    if page == "Home":
        show_home_page()
    elif page == "Single Prediction":
        show_prediction_page()
    elif page == "Batch Analysis":
        show_batch_analysis()
    else:
        show_model_performance()

def show_home_page():
    st.header("Welcome to Spam Email Detector")
    st.write("""
    This application helps you detect spam emails using machine learning.
    Upload your email content or paste it directly to get instant predictions.
    """)
    
    st.subheader("Features")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("- Real-time spam detection")
        st.write("- Detailed analysis visualization")
        
    with col2:
        st.write("- Batch processing support")
        st.write("- Model performance metrics")

def show_prediction_page():
    st.header("Single Email Analysis")
    
    input_method = st.radio(
        "Choose input method",
        ["Text Input", "File Upload"]
    )
    
    if input_method == "Text Input":
        email_content = st.text_area("Enter email content:")
        if email_content:
            process_single_email(email_content)
    else:
        uploaded_file = st.file_uploader("Upload email file", type=["txt"])
        if uploaded_file:
            email_content = uploaded_file.read().decode()
            process_single_email(email_content)

def process_single_email(content):
    st.write("Processing...")
    # Preprocess and show tokens
    tokens = preprocess_email(content)
    st.subheader("Preprocessed Tokens")
    st.write(tokens)

    # Feature extraction
    features = extract_features(content)
    st.subheader("Extracted Features")
    st.json(features)

    # Load model and predict
    model, vectorizer = load_model()
    if model is None:
        st.warning("No trained model found in `models/`. Please train a model or place a saved model at `models/spam_detector.joblib`.")
        return

    prediction, probability = predict(' '.join(tokens), model, vectorizer)
    st.subheader("Prediction")
    st.write(f"Label: {prediction}")
    st.write(f"Probability: {probability}")
    
def show_batch_analysis():
    st.header("Batch Email Analysis")
    # Add batch processing logic here

def show_model_performance():
    st.header("Model Performance Metrics")
    # Add performance visualization logic here

if __name__ == "__main__":
    main()