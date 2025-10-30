"""
Main Streamlit application for spam email detection.
"""
import os
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
    if model is None or vectorizer is None:
        st.warning(
            "No trained model or vectorizer found in `models/`. Please train a model or place the saved artifacts `models/spam_detector.joblib` and `models/spam_detector_vectorizer.joblib`."
        )
        return

    prediction, probability = predict(' '.join(tokens), model, vectorizer)
    st.subheader("Prediction")
    st.write(f"Label: {prediction}")

    # Display probability in a user-friendly way
    if probability is None:
        st.write("Probability: not available for this model")
    elif hasattr(probability, '__iter__'):
        # Binary classification: show probability for positive class if available
        try:
            prob_positive = float(probability[1])
            st.write(f"Probability (spam): {prob_positive:.3f}")
        except Exception:
            # Fallback: show full probability array
            st.write(f"Probability: {list(probability)}")
    else:
        try:
            prob_val = float(probability)
            st.write(f"Probability: {prob_val:.3f}")
        except Exception:
            st.write(f"Probability: {str(probability)}")

    # Show a word cloud or fallback bar chart for the email
    try:
        wc_fig = plot_wordcloud(' '.join(tokens), title='Email Word Cloud')
        st.plotly_chart(wc_fig, use_container_width=True)
    except Exception as e:
        st.error(f"Could not render word cloud: {e}")

    # If the model exposes feature importances, display top features
    try:
        if hasattr(model, 'feature_importances_'):
            # Attempt to get feature names from the vectorizer
            try:
                feature_names = vectorizer.get_feature_names_out()
            except Exception:
                feature_names = None

            if feature_names is not None:
                fi_fig = plot_feature_importance(model, feature_names)
                st.plotly_chart(fi_fig, use_container_width=True)
    except Exception:
        # Non-fatal: skip feature importance if anything goes wrong
        pass
    
def show_batch_analysis():
    st.header("Batch Email Analysis")
    st.write("Upload a CSV with a `text` column to run batch predictions.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"] )
    if not uploaded:
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return

    if 'text' not in df.columns:
        st.error("CSV must contain a `text` column")
        return

    model, vectorizer = load_model()
    if model is None or vectorizer is None:
        st.warning("No trained model/vectorizer found. Please train and add artifacts to `models/`.")
        return

    # Preprocess and predict
    df['tokens'] = df['text'].fillna('').apply(preprocess_email)
    df['text_proc'] = df['tokens'].apply(lambda toks: ' '.join(toks))
    X_vec = vectorizer.transform(df['text_proc'])
    preds = model.predict(X_vec)
    try:
        probs = model.predict_proba(X_vec)
    except Exception:
        probs = None

    df['prediction'] = preds
    if probs is not None:
        # If binary, show probability for positive class
        try:
            df['prob_spam'] = [p[1] for p in probs]
        except Exception:
            df['prob_spam'] = None

    st.write(df[['text', 'prediction', 'prob_spam']].head(50))

    # Offer download
    try:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button('Download results as CSV', data=csv, file_name='batch_predictions.csv')
    except Exception:
        pass

def show_model_performance():
    st.header("Model Performance Metrics")
    model, vectorizer = load_model()
    if model is None or vectorizer is None:
        st.warning("No trained model/vectorizer found. Please train and add artifacts to `models/`.")
        return

    # Load evaluation data (prefer data/data.csv else sample)
    data_path = 'data/data.csv'
    sample_path = 'data/sample_emails.csv'
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    elif os.path.exists(sample_path):
        df = pd.read_csv(sample_path)
    else:
        st.info('No evaluation data found in data/.')
        return

    if 'text' not in df.columns or 'label' not in df.columns:
        st.error('Evaluation data must contain `text` and `label` columns.')
        return

    df['tokens'] = df['text'].fillna('').apply(preprocess_email)
    df['text_proc'] = df['tokens'].apply(lambda toks: ' '.join(toks))
    X_vec = vectorizer.transform(df['text_proc'])
    y_true = df['label'].map({'ham': 0, 'spam': 1}).values
    y_pred = model.predict(X_vec)
    try:
        y_prob = model.predict_proba(X_vec)[:, 1]
    except Exception:
        y_prob = None

    # Confusion matrix
    try:
        cm_fig = plot_confusion_matrix(y_true, y_pred)
        st.plotly_chart(cm_fig, use_container_width=True)
    except Exception as e:
        st.error(f'Could not plot confusion matrix: {e}')

    # ROC
    if y_prob is not None:
        try:
            roc_fig = plot_roc_curve(y_true, y_prob)
            st.plotly_chart(roc_fig, use_container_width=True)
        except Exception as e:
            st.error(f'Could not plot ROC: {e}')

if __name__ == "__main__":
    main()