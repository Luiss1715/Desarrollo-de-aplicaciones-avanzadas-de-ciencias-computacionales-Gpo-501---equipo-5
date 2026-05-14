# Desarrollo-de-aplicaciones-avanzadas-de-ciencias-computacionales-Gpo-501---equipo-5

Pipeline de deteccion de suicidalidad en Python. Clasificacion binaria con evaluacion ROC-AUC.

## Descripcion general

Este repositorio implementa una tuberia modular para detectar riesgo suicida en texto. El flujo es secuencial y cada etapa tiene una responsabilidad clara. Ademas, existe un **fast-path**: si el lexer encuentra frases criticas explicitas, el sistema puede devolver riesgo alto sin invocar el modelo.

### Flujo completo (paso a paso)

1. **Ingesta de datos**
	- Lee texto desde CSV o desde la CLI.
	- Une `title` + `text` en un solo campo.
	- Convierte etiquetas `yes/no` a `1/0` para entrenamiento.

2. **Analisis lexico (lexer)**
	- Escanea el texto crudo con regex basadas en un lexicon curado.
	- Produce:
	  - `tokens`: tokenizacion basica.
	  - `flags`: banderas binarias por categoria del lexicon.
	  - `critical_matches`: frases explicitas de alta alerta.
	- **Fast-path**: si `critical_matches` no esta vacio, se puede devolver `risk_label=high` y `risk_score=1.0`.

3. **Preprocesamiento**
	- Normaliza URLs/emails, pasa a minusculas y colapsa espacios.
	- Opcionalmente aplica lematizacion con spaCy (si esta habilitado) y filtra stopwords.

4. **Extraccion de caracteristicas**
	- Genera TF-IDF con n-gramas de palabra.
	- Concatena las banderas lexicas `flags` al vector TF-IDF para conservar senales basadas en reglas.

5. **Clasificacion**
	- Usa regresion logistica con `class_weight="balanced"`.
	- Devuelve probabilidad (score) y etiqueta binaria.

6. **Salida**
	- Estructura de salida tipo JSON:
	  - `risk_label`: low/high.
	  - `risk_score`: probabilidad en [0,1].
	  - `alert_tokens`: frases criticas detectadas.
	  - `model_version`: version del modelo para reproducibilidad.

### Esquema de datos

- **Entrada CSV**
	- Columnas requeridas: `title`, `text`, `is_suicide`.
	- Etiquetas: `yes` (suicida) / `no` (no suicida).

- **Salida**
	- `risk_label`: low/high.
	- `risk_score`: float.
	- `alert_tokens`: lista de frases criticas.
	- `model_version`: string.

### Archivos y responsabilidades

- `src/suicidality/ingest.py`: lectura de CSV, union de campos y mapeo de etiquetas.
- `src/suicidality/lexing.py`: lexer, reglas del lexicon y fast-path.
- `src/suicidality/preprocess.py`: limpieza y lematizacion opcional.
- `src/suicidality/features.py`: TF-IDF + banderas lexicas.
- `src/suicidality/model.py`: wrapper de regresion logistica.
- `src/suicidality/pipeline.py`: orquestacion de la tuberia.
- `src/suicidality/evaluate.py`: ROC-AUC y curva ROC.
- `src/suicidality/cli.py`: comandos de entrenamiento/evaluacion/prediccion.

### Protocolo de evaluacion

- Split train/test: **80/20 estratificado** con semilla fija.
- Metrica: **ROC-AUC**.
- Artefactos:
	- `reports/metrics.json`
	- `reports/roc.png`

## Instalacion

```bash
pip install -r requirements.txt
```

Opcional: instalar modelos de spaCy si se activa lematizacion.

```bash
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm
```

## Entrenamiento y evaluacion

```bash
python -m suicidality.cli train --csv DataSet.csv
```

Artefactos generados:
- models/pipeline.joblib
- reports/metrics.json
- reports/roc.png

## Evaluar un modelo entrenado

```bash
python -m suicidality.cli eval --csv DataSet.csv --model models/pipeline.joblib
```

## Prediccion de un texto

```bash
python -m suicidality.cli predict --model models/pipeline.joblib --text "I want to die"
```

## Documentos

- docs/parteA.md
- docs/parteB.md