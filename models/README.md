Models directory

Place trained model artifacts here. The training script `src/train.py` saves:
This folder is where trained model artifacts are stored for local testing or deployment. By default, `src/train.py` saves two artifacts:

- `models/spam_detector.joblib` — trained classifier
- `models/spam_detector_vectorizer.joblib` — fitted TfidfVectorizer

Notes and best practices:

- Do not commit large binary model files to the repository. Add `models/` to `.gitignore` and use GitHub Releases or an artifact store for distributions.
- The Streamlit app will look for both the model and the vectorizer. If either is missing the app will warn and skip predictions.
- If you need to ship a lightweight demo, create a small, pre-trained artifact and add clear instructions for how to replace or update these files.

When deploying the Streamlit app, ensure these files are present or the app will show a warning and skip predictions.