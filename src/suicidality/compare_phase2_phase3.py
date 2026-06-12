from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .config import LEXICON_PATH, TrainConfig
from .fase3.embeddings import DEFAULT_EMBEDDING_MODEL, TransformerEmbedder, prepare_texts
from .evaluate import save_roc_curve
from .fase2.features import FeatureExtractor
from .fase3.latent_classifiers import (
    LATENT_CLASSIFIER_NAMES,
    fit_latent_classifier,
    predict_positive_scores,
)
from .fase3.latent_nn import LatentNNConfig, predict_labels, predict_scores, save_latent_model, train_mlp
from .fase3.latent_pipeline import load_phase3_dataset
from .fase3.latent_visualization import plot_projection
from .fase2.lexing import RiskLexer
from .fase2.model import SuicideRiskModel
from .fase3.nli_classifier import DEFAULT_NLI_MODEL, SupervisedNLIClassifier, ZeroShotNLIClassifier
from .fase2.pipeline import SuicidalityPipeline
from .protocol_metrics import (
    build_prediction_frame,
    calculate_protocol_metrics,
    find_best_threshold,
    labels_from_frame,
    save_metrics,
)


COMPARISON_COLUMNS = [
    "method",
    "TP",
    "TN",
    "FP",
    "FN",
    "TPR",
    "FPR",
    "AUC",
    "ROC-AUC",
    "comentario",
]

LATENT_METHODS = {
    "logistic_regression": (
        "phase3_latent_logreg",
        "Embeddings de Transformer y regresion logistica",
    ),
    "linear_svm": (
        "phase3_latent_svm",
        "Embeddings de Transformer y SVM lineal",
    ),
    "random_forest": (
        "phase3_latent_random_forest",
        "Embeddings de Transformer y Random Forest",
    ),
}


def build_phase2_pipeline(config: TrainConfig) -> SuicidalityPipeline:
    return SuicidalityPipeline(
        config=config,
        lexer=RiskLexer(LEXICON_PATH),
        features=FeatureExtractor(
            min_df=config.tfidf_min_df, ngram_range=config.tfidf_ngram_range
        ),
        model=SuicideRiskModel(),
    )


def build_comparison_table(rows: list[dict[str, object]]) -> pd.DataFrame:
    comparison = pd.DataFrame(rows).rename(
        columns={"roc_auc": "ROC-AUC", "notes": "comentario"}
    )
    if "ROC-AUC" not in comparison:
        comparison["ROC-AUC"] = np.nan
    return comparison[COMPARISON_COLUMNS]


def split_internal_validation(
    frame: pd.DataFrame,
    labels: list[int],
    validation_size: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    indices = np.arange(len(frame))
    if "user_id" in frame.columns and frame["user_id"].notna().any():
        groups = frame["user_id"].fillna(frame.index.to_series().map(lambda value: f"row-{value}"))
        splitter = GroupShuffleSplit(n_splits=20, test_size=validation_size, random_state=seed)
        candidates = [
            split
            for split in splitter.split(indices, labels, groups)
            if len(set(np.asarray(labels)[split[0]])) == 2
            and len(set(np.asarray(labels)[split[1]])) == 2
        ]
        if not candidates:
            raise ValueError("Could not create a grouped validation split containing both classes")
        train_indices, validation_indices = min(
            candidates,
            key=lambda split: abs(np.mean(np.asarray(labels)[split[1]]) - np.mean(labels)),
        )
        return train_indices.tolist(), validation_indices.tolist()
    train_indices, validation_indices = train_test_split(
        indices, test_size=validation_size, random_state=seed, stratify=labels
    )
    return train_indices.tolist(), validation_indices.tolist()


def compare_methods(
    csv_path: str | Path,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    nli_model: str = DEFAULT_NLI_MODEL,
    threshold: float = 0.5,
    test_size: float = 0.2,
    seed: int = 42,
    epochs: int = 20,
    batch_size: int = 32,
    reports_dir: str | Path = "reports",
    models_dir: str | Path = "models",
    device: str | None = None,
    nli_device: int | None = None,
    test_csv_path: str | Path | None = None,
) -> pd.DataFrame:
    if test_csv_path is not None and Path(csv_path).resolve() == Path(test_csv_path).resolve():
        raise ValueError("Training and test CSV files must be different")
    frame = load_phase3_dataset(csv_path)
    labels = labels_from_frame(frame)
    texts = prepare_texts(frame)
    internal_train_indices, validation_indices = split_internal_validation(
        frame, labels, test_size, seed
    )

    if test_csv_path is not None:
        test_frame = load_phase3_dataset(test_csv_path)
        actual_test_labels = labels_from_frame(test_frame)
        test_texts = prepare_texts(test_frame)
        test_frame_for_output = test_frame
        test_indices_for_output = list(range(len(test_frame)))
    else:
        test_texts = [texts[index] for index in validation_indices]
        actual_test_labels = [labels[index] for index in validation_indices]
        test_frame_for_output = frame
        test_indices_for_output = validation_indices
    reports_path = Path(reports_dir)
    models_path = Path(models_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    models_path.mkdir(parents=True, exist_ok=True)

    phase2_validation = build_phase2_pipeline(TrainConfig(test_size=test_size, random_seed=seed))
    phase2_validation.fit(
        [texts[index] for index in internal_train_indices],
        [labels[index] for index in internal_train_indices],
    )
    phase2_validation_scores = phase2_validation.predict_proba(
        [texts[index] for index in validation_indices]
    )
    phase2_threshold = find_best_threshold(
        [labels[index] for index in validation_indices], phase2_validation_scores
    )
    phase2 = build_phase2_pipeline(TrainConfig(test_size=test_size, random_seed=seed))
    phase2.fit(texts, labels)
    joblib.dump(phase2, models_path / "pipeline.joblib")
    phase2_scores = phase2.predict_proba(test_texts)
    phase2_predicted = [1 if score >= phase2_threshold else 0 for score in phase2_scores]

    embedder = TransformerEmbedder(embedding_model, device=device)
    embeddings = embedder.encode_chunked(texts, batch_size=batch_size)
    plot_projection(
        PCA(n_components=2).fit_transform(embeddings),
        labels,
        "Latent space: PCA",
        reports_path / "latent_space_pca.png",
    )
    base_latent_config = LatentNNConfig(input_dim=embeddings.shape[1], embedding_model=embedding_model)
    validation_labels = [labels[index] for index in validation_indices]
    latent_validation_model, history = train_mlp(
        embeddings[internal_train_indices],
        [labels[index] for index in internal_train_indices],
        embeddings[validation_indices],
        validation_labels,
        base_latent_config,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        device=device,
        use_class_weight=True,
        weight_decay=1e-4,
    )
    latent_validation_scores = predict_scores(
        latent_validation_model, embeddings[validation_indices], device=device
    )
    best_latent_threshold = find_best_threshold(validation_labels, latent_validation_scores)
    best_epoch = int(max(history, key=lambda row: row["validation_roc_auc"])["epoch"])
    latent_config = LatentNNConfig(
        input_dim=base_latent_config.input_dim,
        embedding_model=base_latent_config.embedding_model,
        hidden_dim_1=base_latent_config.hidden_dim_1,
        hidden_dim_2=base_latent_config.hidden_dim_2,
        dropout=base_latent_config.dropout,
        threshold=best_latent_threshold,
    )
    latent_model, _ = train_mlp(
        embeddings,
        labels,
        embeddings[validation_indices],
        validation_labels,
        latent_config,
        epochs=best_epoch,
        batch_size=batch_size,
        seed=seed,
        device=device,
        use_class_weight=True,
        weight_decay=1e-4,
        early_stopping_patience=None,
        restore_best_weights=False,
    )
    save_latent_model(
        latent_model,
        latent_config,
        models_path / "latent_nn.pt",
        models_path / "latent_config.json",
    )
    test_embeddings = embedder.encode_chunked(test_texts, batch_size=batch_size)
    latent_scores = predict_scores(latent_model, test_embeddings, device=device)
    latent_predicted = predict_labels(latent_scores, latent_config.threshold).tolist()

    latent_classic_results = {}
    for classifier_name in LATENT_CLASSIFIER_NAMES:
        validation_classifier = fit_latent_classifier(
            classifier_name,
            embeddings[internal_train_indices],
            [labels[index] for index in internal_train_indices],
            seed,
        )
        validation_scores = predict_positive_scores(
            validation_classifier, embeddings[validation_indices]
        )
        classifier_threshold = find_best_threshold(validation_labels, validation_scores)
        classifier = fit_latent_classifier(classifier_name, embeddings, labels, seed)
        joblib.dump(classifier, models_path / f"latent_{classifier_name}.joblib")
        classifier_scores = predict_positive_scores(classifier, test_embeddings)
        method_name, notes = LATENT_METHODS[classifier_name]
        latent_classic_results[classifier_name] = {
            "method": method_name,
            "notes": notes,
            "threshold": classifier_threshold,
            "validation_scores": validation_scores,
            "scores": classifier_scores,
            "predicted": [
                1 if score >= classifier_threshold else 0 for score in classifier_scores
            ],
        }

    nli_classifier = ZeroShotNLIClassifier(nli_model, threshold, device=nli_device)
    nli_batch_size = max(1, batch_size // 4)
    _, nli_validation_scores = nli_classifier.predict(
        [texts[index] for index in validation_indices], batch_size=nli_batch_size
    )
    nli_threshold = find_best_threshold(validation_labels, nli_validation_scores)
    _, nli_scores_array = nli_classifier.predict(test_texts, batch_size=nli_batch_size)
    nli_scores = nli_scores_array.tolist()
    nli_predicted = [1 if score >= nli_threshold else 0 for score in nli_scores]

    supervised_nli = SupervisedNLIClassifier(nli_classifier, seed=seed)
    nli_training_features = supervised_nli.extract_features(texts, batch_size=nli_batch_size)
    nli_test_features = supervised_nli.extract_features(test_texts, batch_size=nli_batch_size)
    supervised_nli_validation = SupervisedNLIClassifier(nli_classifier, seed=seed)
    supervised_nli_validation.fit_features(
        nli_training_features[internal_train_indices],
        [labels[index] for index in internal_train_indices],
    )
    _, supervised_nli_validation_scores = supervised_nli_validation.predict_from_features(
        nli_training_features[validation_indices]
    )
    supervised_nli_threshold = find_best_threshold(
        validation_labels, supervised_nli_validation_scores
    )
    supervised_nli.fit_features(nli_training_features, labels)
    _, supervised_nli_scores = supervised_nli.predict_from_features(
        nli_test_features, supervised_nli_threshold
    )
    supervised_nli_predicted = [
        1 if score >= supervised_nli_threshold else 0 for score in supervised_nli_scores
    ]
    joblib.dump(
        {
            "candidate_labels": supervised_nli.candidate_labels,
            "classifier": supervised_nli.classifier,
        },
        models_path / "nli_supervised.joblib",
    )

    validation_stack = np.column_stack(
        [
            latent_validation_scores,
            *[
                latent_classic_results[name]["validation_scores"]
                for name in LATENT_CLASSIFIER_NAMES
            ],
            nli_validation_scores,
            supervised_nli_validation_scores,
        ]
    )
    ensemble_model = LogisticRegression(class_weight="balanced", random_state=seed)
    ensemble_model.fit(validation_stack, validation_labels)
    ensemble_validation_scores = ensemble_model.predict_proba(validation_stack)[:, 1]
    ensemble_threshold = find_best_threshold(validation_labels, ensemble_validation_scores)
    joblib.dump(ensemble_model, models_path / "phase3_ensemble.joblib")
    ensemble_scores = ensemble_model.predict_proba(
        np.column_stack(
            [
                latent_scores,
                *[latent_classic_results[name]["scores"] for name in LATENT_CLASSIFIER_NAMES],
                nli_scores,
                supervised_nli_scores,
            ]
        )
    )[:, 1]
    ensemble_predicted = [1 if score >= ensemble_threshold else 0 for score in ensemble_scores]

    thresholds = {
        "phase2": phase2_threshold,
        "phase3_latent_mlp": best_latent_threshold,
        **{
            str(latent_classic_results[name]["method"]): float(
                latent_classic_results[name]["threshold"]
            )
            for name in LATENT_CLASSIFIER_NAMES
        },
        "phase3_nli_zero_shot": nli_threshold,
        "phase3_nli_supervised": supervised_nli_threshold,
        "phase3_ensemble": ensemble_threshold,
        "latent_best_epoch": best_epoch,
    }
    (models_path / "phase3_thresholds.json").write_text(
        json.dumps(thresholds, indent=2), encoding="utf-8"
    )

    methods = [
        ("phase2", phase2_predicted, phase2_scores, "TF-IDF, banderas lexicas y regresion logistica"),
        (
            "phase3_latent_mlp",
            latent_predicted,
            latent_scores.tolist(),
            "Embeddings de Transformer y MLP",
        ),
        *[
            (
                str(latent_classic_results[name]["method"]),
                latent_classic_results[name]["predicted"],
                latent_classic_results[name]["scores"].tolist(),
                str(latent_classic_results[name]["notes"]),
            )
            for name in LATENT_CLASSIFIER_NAMES
        ],
        ("phase3_nli_zero_shot", nli_predicted, nli_scores, f"NLI zero-shot: {nli_model}"),
        (
            "phase3_nli_supervised",
            supervised_nli_predicted,
            supervised_nli_scores.tolist(),
            "NLI supervisado: regresion logistica sobre puntajes de multiples hipotesis",
        ),
        (
            "phase3_ensemble",
            ensemble_predicted,
            ensemble_scores.tolist(),
            "Stacking calibrado de los seis clasificadores de Fase 3",
        ),
    ]
    comparison_metrics = {}
    table_rows = []
    
    for method, predicted, scores, notes in methods:
        predictions = build_prediction_frame(test_frame_for_output, test_indices_for_output, predicted, scores, method)
        predictions.to_csv(reports_path / f"{method}_predictions.csv", index=False)
        metrics = calculate_protocol_metrics(actual_test_labels, predicted, scores)
        comparison_metrics[method] = metrics
        table_rows.append({**metrics, "method": method, "notes": notes})
        if method == "phase2":
            save_metrics(metrics, reports_path / "metrics.json")
            save_roc_curve(actual_test_labels, scores, reports_path / "roc.png")

    (reports_path / "comparison_metrics.json").write_text(
        json.dumps(comparison_metrics, indent=2), encoding="utf-8"
    )
    comparison = build_comparison_table(table_rows)
    comparison.to_csv(reports_path / "comparison_table.csv", index=False)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Phase 2 and Phase 3 methods")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--nli-model", default=DEFAULT_NLI_MODEL)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--test-csv",
        required=True,
        help="Separate CSV used exclusively for final evaluation",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--device")
    parser.add_argument("--nli-device", type=int)
    args = parser.parse_args()
    compare_methods(
        args.csv,
        embedding_model=args.embedding_model,
        nli_model=args.nli_model,
        threshold=args.threshold,
        test_size=args.test_size,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        reports_dir=args.reports_dir,
        models_dir=args.models_dir,
        device=args.device,
        nli_device=args.nli_device,
        test_csv_path=args.test_csv,
    )


if __name__ == "__main__":
    main()
