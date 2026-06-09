from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import LEXICON_PATH, TrainConfig
from .embeddings import DEFAULT_EMBEDDING_MODEL, TransformerEmbedder, prepare_texts
from .features import FeatureExtractor
from .latent_nn import LatentNNConfig, predict_labels, predict_scores, save_latent_model, train_mlp
from .latent_pipeline import load_phase3_dataset
from .lexing import RiskLexer
from .model import SuicideRiskModel
from .nli_classifier import DEFAULT_NLI_MODEL, ZeroShotNLIClassifier
from .pipeline import SuicidalityPipeline
from .protocol_metrics import build_prediction_frame, calculate_protocol_metrics, labels_from_frame


def build_phase2_pipeline(config: TrainConfig) -> SuicidalityPipeline:
    return SuicidalityPipeline(
        config=config,
        lexer=RiskLexer(LEXICON_PATH),
        features=FeatureExtractor(
            min_df=config.tfidf_min_df, ngram_range=config.tfidf_ngram_range
        ),
        model=SuicideRiskModel(),
    )


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
) -> pd.DataFrame:
    frame = load_phase3_dataset(csv_path)
    labels = labels_from_frame(frame)
    texts = prepare_texts(frame)
    train_indices, test_indices = train_test_split(
        list(range(len(frame))), test_size=test_size, random_state=seed, stratify=labels
    )
    reports_path = Path(reports_dir)
    models_path = Path(models_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    models_path.mkdir(parents=True, exist_ok=True)

    phase2 = build_phase2_pipeline(TrainConfig(test_size=test_size, random_seed=seed))
    phase2.fit([texts[index] for index in train_indices], [labels[index] for index in train_indices])
    joblib.dump(phase2, models_path / "pipeline.joblib")
    phase2_scores = phase2.predict_proba([texts[index] for index in test_indices])
    phase2_predicted = [1 if score >= threshold else 0 for score in phase2_scores]

    embeddings = TransformerEmbedder(embedding_model, device=device).encode(texts, batch_size=batch_size)
    latent_config = LatentNNConfig(
        input_dim=embeddings.shape[1], embedding_model=embedding_model, threshold=threshold
    )
    latent_model, _ = train_mlp(
        embeddings[train_indices],
        [labels[index] for index in train_indices],
        embeddings[test_indices],
        [labels[index] for index in test_indices],
        latent_config,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )
    save_latent_model(
        latent_model,
        latent_config,
        models_path / "latent_nn.pt",
        models_path / "latent_config.json",
    )
    latent_scores = predict_scores(latent_model, embeddings[test_indices], device=device)
    latent_predicted = predict_labels(latent_scores, threshold).tolist()

    nli_predicted_array, nli_scores_array = ZeroShotNLIClassifier(
        nli_model, threshold, device=nli_device
    ).predict([texts[index] for index in test_indices], batch_size=max(1, batch_size // 4))
    nli_predicted = nli_predicted_array.tolist()
    nli_scores = nli_scores_array.tolist()

    methods = [
        ("phase2", phase2_predicted, phase2_scores, "TF-IDF, lexical flags and logistic regression"),
        ("phase3_latent", latent_predicted, latent_scores.tolist(), "Transformer embeddings and MLP"),
        ("phase3_nli", nli_predicted, nli_scores, f"Zero-shot NLI: {nli_model}"),
    ]
    comparison_metrics = {}
    table_rows = []
    test_labels = [labels[index] for index in test_indices]
    for method, predicted, scores, notes in methods:
        predictions = build_prediction_frame(frame, test_indices, predicted, scores, method)
        predictions.to_csv(reports_path / f"{method}_predictions.csv", index=False)
        metrics = calculate_protocol_metrics(test_labels, predicted, scores)
        comparison_metrics[method] = metrics
        table_rows.append({**metrics, "method": method, "notes": notes})

    (reports_path / "comparison_metrics.json").write_text(
        json.dumps(comparison_metrics, indent=2), encoding="utf-8"
    )
    columns = ["method", "TP", "TN", "FP", "FN", "TPR", "FPR", "AUC", "notes"]
    comparison = pd.DataFrame(table_rows)
    comparison[columns].to_csv(reports_path / "comparison_table.csv", index=False)
    return comparison[columns]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Phase 2 and Phase 3 methods")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--nli-model", default=DEFAULT_NLI_MODEL)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--test-size", type=float, default=0.2)
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
    )


if __name__ == "__main__":
    main()
