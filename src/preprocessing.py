"""
Email preprocessing utilities.
"""
import re
"""
Preprocessing utilities that are resilient when NLTK is not available.
This module avoids downloading NLTK data at import time and provides
lightweight fallbacks so tests and CI can run without network access.
"""
import importlib

_nltk = None
_nltk_available = False
try:
    _nltk = importlib.import_module('nltk')
    # Import submodules lazily
    from nltk.corpus import stopwords  # type: ignore
    from nltk.tokenize import word_tokenize  # type: ignore
    from nltk.stem import WordNetLemmatizer  # type: ignore
    _nltk_available = True
except Exception:
    # NLTK not available — we'll use simple fallbacks
    _nltk_available = False

def preprocess_email(text):
    """
    Preprocess email content for spam detection.
    
    Args:
        text (str): Raw email content
        
    Returns:
        list: Preprocessed tokens
    """
    # Normalize
    text = (text or '').strip()
    text_lower = text.lower()

    # Remove email addresses and URLs
    text_clean = re.sub(r"\S+@\S+", "", text_lower)
    text_clean = re.sub(r"http\S+|www\.\S+", "", text_clean)

    # Remove special characters and numbers, keep whitespace
    text_clean = re.sub(r"[^a-zA-Z\s]", " ", text_clean)

    # Tokenization with fallback
    nltk_ok = bool(_nltk_available and _nltk is not None)
    if nltk_ok:
        try:
            # Ensure required NLTK resources are available; download only if missing
            try:
                _ = getattr(_nltk, 'data').find('tokenizers/punkt')
            except Exception:
                getattr(_nltk, 'download')('punkt')
            try:
                _ = getattr(_nltk, 'data').find('corpora/stopwords')
            except Exception:
                getattr(_nltk, 'download')('stopwords')
            try:
                _ = getattr(_nltk, 'data').find('corpora/wordnet')
            except Exception:
                getattr(_nltk, 'download')('wordnet')

            tokens = word_tokenize(text_clean)
            stop_words = set(stopwords.words('english'))
            tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(t) for t in tokens]
        except Exception:
            # If anything goes wrong with NLTK at runtime, fall back
            nltk_ok = False

    if not nltk_ok:
        # Simple fallback tokenizer: extract words
        tokens = re.findall(r"\b[a-zA-Z]+\b", text_clean)
        # Minimal stopword list to avoid a heavy dependency
        _fallback_stopwords = {
            'the', 'and', 'is', 'in', 'it', 'of', 'to', 'a', 'for', 'on', 'that', 'this'
        }
        tokens = [t for t in tokens if t not in _fallback_stopwords]

    return tokens

def extract_features(text):
    """
    Extract features from preprocessed email text.
    
    Args:
        text (str): Preprocessed email content
        
    Returns:
        dict: Feature dictionary
    """
    txt = text or ''
    length = len(txt)
    word_count = len(re.findall(r"\b\w+\b", txt))
    uppercase_chars = sum(1 for c in txt if c.isupper())
    uppercase_ratio = uppercase_chars / max(1, length)

    features = {
        'length': length,
        'word_count': word_count,
        'uppercase_ratio': uppercase_ratio,
        # Add more feature extraction logic here
    }
    return features