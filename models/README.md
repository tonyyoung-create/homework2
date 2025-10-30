Models directory

Place trained model artifacts here. The training script `src/train.py` saves:
- `models/spam_detector.joblib` (model)
- `models/spam_detector_vectorizer.joblib` (tf-idf vectorizer)

When deploying the Streamlit app, ensure these files are present or the app will show a warning and skip predictions.