from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

from .config import LEXICON_PATH, TrainConfig
from .evaluate import evaluate_auc, save_metrics, save_roc_curve
from .features import FeatureExtractor
from .ingest import load_dataset
from .lexing import RiskLexer
from .model import SuicideRiskModel
from .pipeline import SuicidalityPipeline


def _build_pipeline(config: TrainConfig) -> SuicidalityPipeline:
    lexer = RiskLexer(LEXICON_PATH)
    features = FeatureExtractor(min_df=config.tfidf_min_df, ngram_range=config.tfidf_ngram_range)
    model = SuicideRiskModel()
    return SuicidalityPipeline(config=config, lexer=lexer, features=features, model=model)


def _save_model(pipeline: SuicidalityPipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def _load_model(path: Path) -> SuicidalityPipeline:
    return joblib.load(path)


def cmd_train(args: argparse.Namespace) -> None:
    config = TrainConfig(test_size=args.test_size, random_seed=args.seed, use_spacy=args.use_spacy)
    dataset = load_dataset(args.csv)
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.texts,
        dataset.labels,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=dataset.labels,
    )
    pipeline = _build_pipeline(config)
    pipeline.fit(x_train, y_train)
    _save_model(pipeline, Path(args.model_out))

    scores = pipeline.predict_proba(x_test)
    result = evaluate_auc(y_test, scores)
    save_metrics(result, Path(args.metrics_out))
    save_roc_curve(y_test, scores, Path(args.roc_out))


def cmd_eval(args: argparse.Namespace) -> None:
    dataset = load_dataset(args.csv)
    pipeline = _load_model(Path(args.model))
    scores = pipeline.predict_proba(dataset.texts)
    result = evaluate_auc(dataset.labels, scores)
    save_metrics(result, Path(args.metrics_out))
    save_roc_curve(dataset.labels, scores, Path(args.roc_out))


def cmd_predict(args: argparse.Namespace) -> None:
    pipeline = _load_model(Path(args.model))
    label, score, alerts = pipeline.predict_text(args.text)
    print({"label": int(label), "score": float(score), "alerts": alerts})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Suicidality detection CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train and evaluate")
    train_p.add_argument("--csv", required=True)
    train_p.add_argument("--model-out", default="models/pipeline.joblib")
    train_p.add_argument("--metrics-out", default="reports/metrics.json")
    train_p.add_argument("--roc-out", default="reports/roc.png")
    train_p.add_argument("--test-size", type=float, default=0.2)
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--use-spacy", action="store_true")
    train_p.set_defaults(func=cmd_train)

    eval_p = sub.add_parser("eval", help="Evaluate a trained model")
    eval_p.add_argument("--csv", required=True)
    eval_p.add_argument("--model", required=True)
    eval_p.add_argument("--metrics-out", default="reports/metrics.json")
    eval_p.add_argument("--roc-out", default="reports/roc.png")
    eval_p.set_defaults(func=cmd_eval)

    pred_p = sub.add_parser("predict", help="Predict a single text")
    pred_p.add_argument("--model", required=True)
    pred_p.add_argument("--text", required=True)
    pred_p.set_defaults(func=cmd_predict)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
