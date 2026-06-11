from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from .config import LEXICON_PATH, TrainConfig
from .fase3.embeddings import DEFAULT_EMBEDDING_MODEL, TransformerEmbedder, prepare_texts
from .evaluate import save_roc_curve
from .fase2.features import FeatureExtractor
from .fase3.latent_nn import LatentNNConfig, predict_labels, predict_scores, save_latent_model, train_mlp
from .fase3.latent_pipeline import load_phase3_dataset
from .fase3.latent_visualization import plot_projection
from .fase2.lexing import RiskLexer
from .fase2.model import SuicideRiskModel
from .fase3.nli_classifier import DEFAULT_NLI_MODEL, ZeroShotNLIClassifier
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
    frame = load_phase3_dataset(csv_path)
    labels = labels_from_frame(frame)
    texts = prepare_texts(frame)
    
    if test_csv_path is not None:
        # Use separate test set
        test_frame = load_phase3_dataset(test_csv_path)
        test_labels = labels_from_frame(test_frame)
        test_texts = prepare_texts(test_frame)
        train_indices = list(range(len(frame)))
        test_indices = None  # Not used with separate test set
    else:
        # Split train/test from same dataset
        train_indices, test_indices = train_test_split(
            list(range(len(frame))), test_size=test_size, random_state=seed, stratify=labels
        )
        test_texts = None
        test_labels = None
    reports_path = Path(reports_dir)
    models_path = Path(models_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    models_path.mkdir(parents=True, exist_ok=True)

    phase2 = build_phase2_pipeline(TrainConfig(test_size=test_size, random_seed=seed))
    phase2.fit([texts[index] for index in train_indices], [labels[index] for index in train_indices])
    joblib.dump(phase2, models_path / "pipeline.joblib")
    
    # Get test texts and labels based on whether separate test set was provided
    if test_csv_path is not None:
        phase2_test_texts = test_texts
        actual_test_labels = test_labels
    else:
        phase2_test_texts = [texts[index] for index in test_indices]
        actual_test_labels = [labels[index] for index in test_indices]
    
    phase2_scores = phase2.predict_proba(phase2_test_texts)
    phase2_predicted = [1 if score >= threshold else 0 for score in phase2_scores]
    phase2_metrics = calculate_protocol_metrics(actual_test_labels, phase2_predicted, phase2_scores)
    save_metrics(phase2_metrics, reports_path / "metrics.json")
    save_roc_curve(actual_test_labels, phase2_scores, reports_path / "roc.png")

    embeddings = TransformerEmbedder(embedding_model, device=device).encode(texts, batch_size=batch_size)
    plot_projection(
        PCA(n_components=2).fit_transform(embeddings),
        labels,
        "Latent space: PCA",
        reports_path / "latent_space_pca.png",
    )
    latent_config = LatentNNConfig(
        input_dim=embeddings.shape[1], embedding_model=embedding_model, threshold=threshold
    )
    
    # Prepare test embeddings and labels for latent model training
    if test_csv_path is not None:
        test_embeddings = TransformerEmbedder(embedding_model, device=device).encode(test_texts, batch_size=batch_size)
        latent_test_labels = test_labels
    else:
        test_embeddings = embeddings[test_indices]
        latent_test_labels = [labels[index] for index in test_indices]
    
    latent_model, _ = train_mlp(
        embeddings[train_indices],
        [labels[index] for index in train_indices],
        test_embeddings,
        latent_test_labels,
        latent_config,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        device=device,
        use_class_weight=True,
        weight_decay=1e-5,
    )
    latent_train_scores = predict_scores(latent_model, embeddings[train_indices], device=device)
    best_latent_threshold = find_best_threshold(
        [labels[index] for index in train_indices],
        latent_train_scores,
    )
    latent_config = LatentNNConfig(
        input_dim=latent_config.input_dim,
        embedding_model=latent_config.embedding_model,
        hidden_dim_1=latent_config.hidden_dim_1,
        hidden_dim_2=latent_config.hidden_dim_2,
        dropout=latent_config.dropout,
        threshold=best_latent_threshold,
    )
    save_latent_model(
        latent_model,
        latent_config,
        models_path / "latent_nn.pt",
        models_path / "latent_config.json",
    )
    latent_scores = predict_scores(latent_model, test_embeddings, device=device)
    latent_predicted = predict_labels(latent_scores, latent_config.threshold).tolist()

    nli_predicted_array, nli_scores_array = ZeroShotNLIClassifier(
        nli_model, threshold, device=nli_device
    ).predict(test_texts, batch_size=max(1, batch_size // 4))
    nli_predicted = nli_predicted_array.tolist()
    nli_scores = nli_scores_array.tolist()

    ensemble_scores = (
        0.35 * np.asarray(phase2_scores)
        + 0.35 * np.asarray(latent_scores)
        + 0.30 * np.asarray(nli_scores)
    )
    ensemble_predicted = [1 if score >= threshold else 0 for score in ensemble_scores]

    methods = [
        ("phase2", phase2_predicted, phase2_scores, "TF-IDF, banderas lexicas y regresion logistica"),
        ("phase3_latent", latent_predicted, latent_scores.tolist(), "Embeddings de Transformer y MLP"),
        ("phase3_nli", nli_predicted, nli_scores, f"NLI zero-shot: {nli_model}"),
        (
            "phase3_ensemble",
            ensemble_predicted,
            ensemble_scores.tolist(),
            "Ensamble de TF-IDF, embeddings y NLI",
        ),
    ]
    comparison_metrics = {}
    table_rows = []
    
    # Prepare test frame and indices for output
    if test_csv_path is not None:
        test_frame_for_output = load_phase3_dataset(test_csv_path)
        test_indices_for_output = list(range(len(test_frame_for_output)))
    else:
        test_frame_for_output = frame
        test_indices_for_output = test_indices
    
    for method, predicted, scores, notes in methods:
        predictions = build_prediction_frame(test_frame_for_output, test_indices_for_output, predicted, scores, method)
        predictions.to_csv(reports_path / f"{method}_predictions.csv", index=False)
        metrics = calculate_protocol_metrics(actual_test_labels, predicted, scores)
        comparison_metrics[method] = metrics
        table_rows.append({**metrics, "method": method, "notes": notes})

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
    parser.add_argument("--test-csv", default=None, help="Optional separate test CSV file")
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
