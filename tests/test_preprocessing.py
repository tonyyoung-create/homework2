from src.preprocessing import preprocess_email, extract_features


def test_preprocess_email_basic():
    text = "Hello user@example.com! Visit http://example.com now. THIS is SPAM!!!"
    tokens = preprocess_email(text)
    assert isinstance(tokens, list)
    # tokens should contain at least one lowercase word
    assert any(t.isalpha() for t in tokens)


def test_extract_features():
    text = "This is a Test Email!"
    features = extract_features(text)
    assert 'length' in features
    assert 'word_count' in features
    assert 'uppercase_ratio' in features
    assert features['word_count'] >= 1
