# Desarrollo-de-aplicaciones-avanzadas-de-ciencias-computacionales-Gpo-501---equipo-5

Suicidality detection pipeline in Python. Binary classification with ROC-AUC evaluation.

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