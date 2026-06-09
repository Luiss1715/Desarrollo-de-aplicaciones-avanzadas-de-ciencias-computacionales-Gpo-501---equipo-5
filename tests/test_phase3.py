import pandas as pd

from suicidality.embeddings import combine_title_text, prepare_texts
from suicidality.nli_classifier import ZeroShotNLIClassifier
from suicidality.protocol_metrics import PREDICTION_COLUMNS, build_prediction_frame


def test_prepare_texts_combines_title_and_text():
    frame = pd.DataFrame({"title": ["A title", None], "text": ["body", "only body"]})

    assert combine_title_text("A title", "body") == "A title body"
    assert prepare_texts(frame) == ["A title body", "only body"]


def test_prediction_frame_has_expected_columns():
    source = pd.DataFrame(
        {
            "title": ["title"],
            "text": ["text"],
            "is_suicide": ["yes"],
        }
    )

    result = build_prediction_frame(source, [0], [1], [0.9], "test_method")

    assert result.columns.tolist() == PREDICTION_COLUMNS
    assert result.loc[0, "predicted_label"] == "yes"
    assert result.loc[0, "method"] == "test_method"


def test_nli_predict_uses_mock_without_loading_model():
    classifier = ZeroShotNLIClassifier(threshold=0.5)
    classifier._classifier = lambda texts, **kwargs: [
        {
            "labels": ["suicidal ideation", "no suicidal ideation"],
            "scores": [0.8, 0.2],
        }
        for _ in texts
    ]

    predicted, scores = classifier.predict(["sample text"])

    assert predicted.tolist() == [1]
    assert scores.tolist() == [0.8]
