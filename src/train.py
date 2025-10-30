"""
Training script for spam detection following a CRISP-DM flow.
Run: python src/train.py
It will:
 - load data/data.csv if present, otherwise use data/sample_emails.csv
 - run simple EDA prints
 - preprocess text using src.preprocessing.preprocess_email
 - vectorize with TfidfVectorizer
 - train RandomForestClassifier
 - evaluate and print metrics
 - save model and vectorizer to models/
"""
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from src.preprocessing import preprocess_email

DATA_PATH = os.path.join('data', 'data.csv')
SAMPLE_PATH = os.path.join('data', 'sample_emails.csv')
MODELS_DIR = 'models'
MODEL_PATH = os.path.join(MODELS_DIR, 'spam_detector.joblib')
VECTORIZER_PATH = os.path.join(MODELS_DIR, 'spam_detector_vectorizer.joblib')


def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        df = pd.read_csv(SAMPLE_PATH)
    return df


def text_join_tokens(tokens):
    return ' '.join(tokens)


def run():
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = load_data()
    print('Data sample:')
    print(df.head())
    print('\nClass distribution:')
    print(df['label'].value_counts())

    # Preprocess
    df['tokens'] = df['text'].fillna('').apply(preprocess_email)
    df['text_proc'] = df['tokens'].apply(text_join_tokens)

    X = df['text_proc']
    y = df['label'].map({'ham': 0, 'spam': 1})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(max_features=2000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    try:
        y_prob = model.predict_proba(X_test_vec)[:, 1]
        roc = roc_auc_score(y_test, y_prob)
    except Exception:
        roc = None

    print('\nClassification report:')
    print(classification_report(y_test, y_pred))
    print('\nConfusion matrix:')
    print(confusion_matrix(y_test, y_pred))
    if roc is not None:
        print(f'ROC AUC: {roc:.3f}')

    # Save artifacts
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f'Artifacts saved: {MODEL_PATH}, {VECTORIZER_PATH}')


if __name__ == '__main__':
    run()
