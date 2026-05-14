# Desarrollo-de-aplicaciones-avanzadas-de-ciencias-computacionales-Gpo-501---equipo-5

Suicidality detection pipeline in Python. Binary classification with ROC-AUC evaluation.

## Overview

This project implements a modular pipeline to detect suicidal ideation from text. The flow is a strict sequence of stages, with a fast-path that can raise a high-risk alert when explicit phrases are found.

### Pipeline behavior (step-by-step)

1. **Ingestion**
	- Reads text from CSV or CLI input.
	- Merges `title` + `text` into a single string.
	- Maps labels `yes/no` to `1/0` for training.

2. **Lexical analysis (lexer)**
	- Scans the raw text with regex patterns from a curated lexicon.
	- Produces:
	  - `tokens` (basic tokenization)
	  - `flags` (binary features per lexicon category)
	  - `critical_matches` (explicit phrases)
	- **Fast-path**: if `critical_matches` is non-empty, the system can return high risk without running the ML model.

3. **Preprocessing**
	- Normalizes URLs/emails, lowercases, and collapses whitespace.
	- Optionally lemmatizes with spaCy (if enabled) and removes stopwords.

4. **Feature extraction**
	- Builds TF-IDF features from the preprocessed text.
	- Concatenates lexical `flags` to the TF-IDF vector to preserve rule-based signals.

5. **Classification**
	- Uses Logistic Regression with `class_weight="balanced"`.
	- Outputs a probability (risk score) and a binary label.

6. **Output**
	- Produces a JSON-like result:
	  - `risk_label` (low/high)
	  - `risk_score` (probability)
	  - `alert_tokens` (critical phrases from the lexer)
	  - `model_version` (for reproducibility)

### Files and responsibilities

- `src/suicidality/ingest.py`: CSV reading and label mapping.
- `src/suicidality/lexing.py`: lexer, lexicon matching, fast-path signals.
- `src/suicidality/preprocess.py`: cleaning + optional lemmatization.
- `src/suicidality/features.py`: TF-IDF + lexical flags.
- `src/suicidality/model.py`: classifier wrapper.
- `src/suicidality/pipeline.py`: orchestration of the stages.
- `src/suicidality/evaluate.py`: ROC-AUC + ROC curve output.
- `src/suicidality/cli.py`: training/evaluation/prediction commands.

### Evaluation protocol

- Train/test split: **80/20 stratified** with a fixed seed.
- Metric: **ROC-AUC**.
- Outputs:
  - `reports/metrics.json`
  - `reports/roc.png`

## Setup

```bash
pip install -r requirements.txt
```

Optional: install spaCy models if you enable lemmatization.

```bash
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm
```

## Train and evaluate

```bash
python -m suicidality.cli train --csv DataSet.csv
```

Artifacts:
- models/pipeline.joblib
- reports/metrics.json
- reports/roc.png

## Evaluate a trained model

```bash
python -m suicidality.cli eval --csv DataSet.csv --model models/pipeline.joblib
```

## Predict a single text

```bash
python -m suicidality.cli predict --model models/pipeline.joblib --text "I want to die"
```

## Docs

- docs/parteA.md
- docs/parteB.md