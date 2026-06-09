from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .embeddings import DEFAULT_EMBEDDING_MODEL, TransformerEmbedder, prepare_texts, save_embeddings
from .latent_nn import (
    LatentNNConfig,
    load_latent_model,
    predict_labels,
    predict_scores,
    save_latent_model,
    train_mlp,
)
from .protocol_metrics import (
    build_prediction_frame,
    calculate_protocol_metrics,
    labels_from_frame,
    save_metrics,
)


def load_phase3_dataset(csv_path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    missing = [column for column in ["title", "text", "is_suicide"] if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    return frame


def train_latent_pipeline(
    frame: pd.DataFrame,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    model_path: str | Path = "models/latent_nn.pt",
    config_path: str | Path = "models/latent_config.json",
    predictions_path: str | Path = "reports/phase3_latent_predictions.csv",
    metrics_path: str | Path = "reports/phase3_latent_metrics.json",
    embeddings_path: str | Path | None = None,
    test_size: float = 0.2,
    seed: int = 42,
    epochs: int = 20,
    batch_size: int = 32,
    device: str | None = None,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    labels = labels_from_frame(frame)
    indices = list(range(len(frame)))
    train_indices, validation_indices = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=labels
    )
    embedder = TransformerEmbedder(model_name, device=device)
    embeddings = embedder.encode(prepare_texts(frame), batch_size=batch_size)
    if embeddings_path:
        save_embeddings(embeddings, embeddings_path)

    config = LatentNNConfig(input_dim=embeddings.shape[1], embedding_model=model_name)
    model, _ = train_mlp(
        embeddings[train_indices],
        [labels[index] for index in train_indices],
        embeddings[validation_indices],
        [labels[index] for index in validation_indices],
        config,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )
    save_latent_model(model, config, model_path, config_path)

    scores = predict_scores(model, embeddings[validation_indices], device=device)
    predicted = predict_labels(scores, config.threshold)
    predictions = build_prediction_frame(
        frame, validation_indices, predicted, scores, "phase3_latent"
    )
    metrics = calculate_protocol_metrics(
        [labels[index] for index in validation_indices], predicted.tolist(), scores.tolist()
    )
    _save_predictions(predictions, predictions_path)
    save_metrics(metrics, metrics_path)
    return predictions, metrics


def evaluate_latent_pipeline(
    frame: pd.DataFrame,
    model_path: str | Path,
    config_path: str | Path,
    predictions_path: str | Path = "reports/phase3_latent_predictions.csv",
    metrics_path: str | Path = "reports/phase3_latent_metrics.json",
    batch_size: int = 32,
    device: str | None = None,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    model, config = load_latent_model(model_path, config_path, device=device)
    embeddings = TransformerEmbedder(config.embedding_model, device=device).encode(
        prepare_texts(frame), batch_size=batch_size
    )
    scores = predict_scores(model, embeddings, device=device)
    predicted = predict_labels(scores, config.threshold)
    labels = labels_from_frame(frame)
    predictions = build_prediction_frame(frame, range(len(frame)), predicted, scores, "phase3_latent")
    metrics = calculate_protocol_metrics(labels, predicted.tolist(), scores.tolist())
    _save_predictions(predictions, predictions_path)
    save_metrics(metrics, metrics_path)
    return predictions, metrics


def _save_predictions(frame: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transformer embeddings and MLP classifier")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the latent-space classifier")
    train_parser.add_argument("--csv", required=True)
    train_parser.add_argument("--model-name", default=DEFAULT_EMBEDDING_MODEL)
    train_parser.add_argument("--model-out", default="models/latent_nn.pt")
    train_parser.add_argument("--config-out", default="models/latent_config.json")
    train_parser.add_argument("--predictions-out", default="reports/phase3_latent_predictions.csv")
    train_parser.add_argument("--metrics-out", default="reports/phase3_latent_metrics.json")
    train_parser.add_argument("--embeddings-out")
    train_parser.add_argument("--test-size", type=float, default=0.2)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--device")

    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained latent-space classifier")
    eval_parser.add_argument("--csv", required=True)
    eval_parser.add_argument("--model", required=True)
    eval_parser.add_argument("--config", default="models/latent_config.json")
    eval_parser.add_argument("--predictions-out", default="reports/phase3_latent_predictions.csv")
    eval_parser.add_argument("--metrics-out", default="reports/phase3_latent_metrics.json")
    eval_parser.add_argument("--batch-size", type=int, default=32)
    eval_parser.add_argument("--device")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    frame = load_phase3_dataset(args.csv)
    if args.command == "train":
        train_latent_pipeline(
            frame,
            model_name=args.model_name,
            model_path=args.model_out,
            config_path=args.config_out,
            predictions_path=args.predictions_out,
            metrics_path=args.metrics_out,
            embeddings_path=args.embeddings_out,
            test_size=args.test_size,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
        )
    else:
        evaluate_latent_pipeline(
            frame,
            model_path=args.model,
            config_path=args.config,
            predictions_path=args.predictions_out,
            metrics_path=args.metrics_out,
            batch_size=args.batch_size,
            device=args.device,
        )


if __name__ == "__main__":
    main()
