import numpy as np
import pandas as pd

from suicidality.compare_phase2_phase3 import (
    COMPARISON_COLUMNS,
    LATENT_METHODS,
    build_comparison_table,
    split_internal_validation,
)
from suicidality.fase3.embeddings import combine_title_text, prepare_texts, split_text_chunks
from suicidality.fase3.latent_classifiers import (
    LATENT_CLASSIFIER_NAMES,
    fit_latent_classifier,
    predict_positive_scores,
)
from suicidality.llm_prompt_classifier import PromptLLMClassifier, parse_prompt_response
from suicidality.fase3.nli_classifier import SupervisedNLIClassifier, ZeroShotNLIClassifier
from suicidality.protocol_metrics import PREDICTION_COLUMNS, build_prediction_frame


def test_prepare_texts_combines_title_and_text():
    frame = pd.DataFrame({"title": ["A title", None], "text": ["body", "only body"]})

    assert combine_title_text("A title", "body") == "A title body"
    assert prepare_texts(frame) == ["A title body", "only body"]


def test_split_text_chunks_uses_overlap():
    chunks = split_text_chunks("one two three four five six", chunk_words=4, overlap_words=2)

    assert chunks == ["one two three four", "three four five six", "five six"]


def test_internal_validation_keeps_users_separate():
    frame = pd.DataFrame({"user_id": ["a", "a", "b", "b", "c", "c", "d", "d"]})
    labels = [0, 0, 1, 1, 0, 0, 1, 1]

    train_indices, validation_indices = split_internal_validation(frame, labels, 0.5, 42)

    train_users = set(frame.iloc[train_indices]["user_id"])
    validation_users = set(frame.iloc[validation_indices]["user_id"])
    assert train_users.isdisjoint(validation_users)


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


def test_comparison_table_has_protocol_and_roc_auc_columns():
    result = build_comparison_table(
        [
            {
                "method": "phase2",
                "TP": 1,
                "TN": 1,
                "FP": 0,
                "FN": 0,
                "TPR": 1.0,
                "FPR": 0.0,
                "AUC": 1.0,
                "ROC-AUC": 1.0,
                "notes": "test",
            }
        ]
    )

    assert result.columns.tolist() == COMPARISON_COLUMNS
    assert result.loc[0, "ROC-AUC"] == 1.0


def test_prompt_classifier_uses_injected_generator():
    classifier = PromptLLMClassifier(
        lambda prompt: '{"label": "yes", "score": 0.8}' if "sample" in prompt else ""
    )

    predicted, scores = classifier.predict(["sample text"])

    assert predicted.tolist() == [1]
    assert scores.tolist() == [0.8]
    assert parse_prompt_response('Result: {"label": "no", "score": 0.2}') == (0, 0.2)


def test_find_best_threshold_returns_best_cutoff():
    y_true = [1, 1, 0, 0]
    y_score = [0.9, 0.8, 0.2, 0.1]
    from suicidality.protocol_metrics import find_best_threshold

    threshold = find_best_threshold(y_true, y_score)

    assert threshold in {0.2, 0.21, 0.22, 0.23, 0.24}


def test_nli_predict_average_templates():
    classifier = ZeroShotNLIClassifier(threshold=0.5, hypothesis_templates=[
        "This text expresses {}.",
        "This text is {}.",
    ])
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


def test_nli_uses_maximum_chunk_score():
    classifier = ZeroShotNLIClassifier(
        threshold=0.5,
        hypothesis_templates=["This text expresses {}."],
        chunk_words=2,
        chunk_overlap=0,
    )
    classifier._classifier = lambda texts, **kwargs: [
        {
            "labels": ["suicidal ideation", "no suicidal ideation"],
            "scores": [score, 1 - score],
        }
        for score in [0.2, 0.9]
    ]

    predicted, scores = classifier.predict(["one two three four"])

    assert predicted.tolist() == [1]
    assert scores.tolist() == [0.9]


def test_all_classic_latent_classifiers_return_probability_like_scores():
    embeddings = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.2],
            [0.9, 0.8],
            [1.0, 1.0],
            [0.2, 0.1],
            [0.8, 0.9],
        ]
    )
    labels = [0, 0, 1, 1, 0, 1]

    for name in LATENT_CLASSIFIER_NAMES:
        classifier = fit_latent_classifier(name, embeddings, labels, seed=7)
        scores = predict_positive_scores(classifier, embeddings)

        assert scores.shape == (len(labels),)
        assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_supervised_nli_learns_from_precomputed_hypothesis_scores():
    features = np.asarray(
        [
            [0.9, 0.8, 0.1],
            [0.8, 0.7, 0.2],
            [0.1, 0.2, 0.9],
            [0.2, 0.1, 0.8],
        ]
    )
    classifier = SupervisedNLIClassifier(candidate_labels=["risk", "harm", "safe"])

    classifier.fit_features(features, [1, 1, 0, 0])
    predicted, scores = classifier.predict_from_features(features)

    assert predicted.tolist() == [1, 1, 0, 0]
    assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_nli_feature_extraction_preserves_requested_label_order():
    classifier = ZeroShotNLIClassifier(hypothesis_templates=["This text expresses {}."])
    classifier._classifier = lambda texts, candidate_labels, **kwargs: [
        {
            "labels": list(reversed(candidate_labels)),
            "scores": [0.2, 0.8],
        }
        for _ in texts
    ]

    features = classifier.predict_features(["sample text"], ["risk", "safe"])

    assert features.tolist() == [[0.8, 0.2]]


def test_phase3_exposes_seven_comparison_methods():
    phase3_methods = {
        "phase3_latent_mlp",
        *(method for method, _ in LATENT_METHODS.values()),
        "phase3_nli_zero_shot",
        "phase3_nli_supervised",
        "phase3_ensemble",
    }

    assert len(phase3_methods) == 7
