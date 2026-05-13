from suicidality.preprocess import preprocess_text


def test_preprocess_basic():
    text = "Visit https://example.com and email me@test.com"
    cleaned = preprocess_text(text, use_spacy=False)
    assert "http" not in cleaned
    assert "@" not in cleaned
