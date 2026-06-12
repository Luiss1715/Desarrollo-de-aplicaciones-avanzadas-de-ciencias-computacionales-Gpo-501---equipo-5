# Desarrollo-de-aplicaciones-avanzadas-de-ciencias-computacionales-Gpo-501---equipo-5

Pipeline de deteccion de suicidalidad en Python. Clasificacion binaria con evaluacion ROC-AUC.

## Descripcion general

Este repositorio implementa una tuberia modular para detectar riesgo suicida en texto. El flujo es secuencial y cada etapa tiene una responsabilidad clara. Las frases criticas detectadas por el lexer se usan como caracteristicas del modelo, pero no determinan automaticamente la clasificacion.

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
	- Las coincidencias criticas se incorporan como banderas lexicas y la clasificacion siempre la realiza el modelo.

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
- `src/suicidality/lexing.py`: lexer y reglas del lexicon.
- `src/suicidality/preprocess.py`: limpieza y lematizacion opcional.
- `src/suicidality/features.py`: TF-IDF + banderas lexicas.
- `src/suicidality/model.py`: wrapper de regresion logistica.
- `src/suicidality/pipeline.py`: orquestacion de la tuberia.
- `src/suicidality/evaluate.py`: ROC-AUC y curva ROC.
- `src/suicidality/cli.py`: comandos de entrenamiento/evaluacion/prediccion.

### Protocolo de evaluacion

- Split train/test: **80/20 estratificado** con semilla fija.
- Metricas: TP, TN, FP, FN, TPR, FPR, AUC del protocolo y ROC-AUC.
- AUC del protocolo: `(1 + TPR - FPR) / 2`.
- Artefactos:
	- `reports/metrics.json`
	- `reports/roc.png`

## Instalacion

```bash
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` permite ejecutar los modulos `suicidality` sin configurar
`PYTHONPATH`.

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
- docs/fase3_nli_llm.md
- docs/fase3_clasificadores.md

## Fase 3: Espacio latente, red neuronal y NLI

La Fase 3 agrega dos métodos sin reemplazar el pipeline de Fase 2:

- Embeddings de Transformer con `sentence-transformers`, usados como espacio
  latente para entrenar una MLP en PyTorch.
- Clasificación zero-shot NLI con un modelo MNLI y umbral configurable.

Las dependencias de modelos se cargan al ejecutar cada comando. La primera
ejecución descarga el modelo seleccionado desde Hugging Face.

### Entrenar el clasificador sobre embeddings

```bash
python -m suicidality.fase3.latent_pipeline train --csv DataSet.csv --model-name sentence-transformers/all-MiniLM-L6-v2
```

Este comando guarda `models/latent_nn.pt`, `models/latent_config.json`, métricas
y predicciones de validación.

### Evaluar el clasificador sobre embeddings

```bash
python -m suicidality.fase3.latent_pipeline eval --csv DataSet.csv --model models/latent_nn.pt
```

### Ejecutar NLI zero-shot

```bash
python -m suicidality.fase3.nli_classifier eval --csv DataSet.csv --model-name facebook/bart-large-mnli --threshold 0.5
```

### Comparar Fase 2 y Fase 3

```bash
python -m suicidality.compare_phase2_phase3 --csv data_train.csv --test-csv data_test_fold2.csv --reports-dir reports --models-dir models
```

La comparacion incluye siete clasificadores de Fase 3: MLP, regresion logistica,
SVM lineal y Random Forest sobre el mismo espacio latente; NLI zero-shot; NLI
supervisado sobre multiples hipotesis; y un ensamble de los seis modelos base.
`phase2` se conserva como linea base adicional, por lo que la tabla final tiene
ocho filas. Consulta `docs/fase3_clasificadores.md` para los detalles.

La comparación reserva una validación interna por `user_id` dentro de
`data_train.csv` para calibrar umbrales, early stopping y el ensamble. Después
reentrena los modelos base con todo `data_train.csv`; `data_test_fold2.csv` se
usa exclusivamente para la evaluación final. Los textos largos se procesan por
fragmentos para evitar perder señales ubicadas al final.

- `reports/phase2_predictions.csv`
- `reports/phase3_latent_mlp_predictions.csv`
- `reports/phase3_latent_logreg_predictions.csv`
- `reports/phase3_latent_svm_predictions.csv`
- `reports/phase3_latent_random_forest_predictions.csv`
- `reports/phase3_nli_zero_shot_predictions.csv`
- `reports/phase3_nli_supervised_predictions.csv`
- `reports/phase3_ensemble_predictions.csv`
- `reports/comparison_metrics.json`
- `reports/comparison_table.csv`
- `reports/metrics.json`
- `reports/roc.png`
- `reports/latent_space_pca.png`
- `models/phase3_thresholds.json`
- `models/phase3_ensemble.joblib`
- `models/latent_logistic_regression.joblib`
- `models/latent_linear_svm.joblib`
- `models/latent_random_forest.joblib`
- `models/nli_supervised.joblib`

La AUC principal se calcula con la fórmula del protocolo:
`AUC = (1 + TPR - FPR) / 2`.

La tabla comparativa incluye tambien ROC-AUC, calculada con los scores continuos.
La primera ejecucion descarga los modelos de Hugging Face y puede requerir varios
minutos, memoria suficiente y conexion a internet.

En un equipo con PyTorch y CUDA:

```bash
python -m suicidality.compare_phase2_phase3 --csv data_train.csv --test-csv data_test_fold2.csv --reports-dir reports --models-dir models --device cuda --nli-device 0
```

### Visualizar el espacio latente

```bash
python -m suicidality.fase3.latent_visualization --csv DataSet.csv --model-name sentence-transformers/all-MiniLM-L6-v2
```

Se genera `reports/latent_space_pca.png`. Si `umap-learn` está instalado,
también se genera `reports/latent_space_umap.png`.

Instalacion opcional de UMAP:

```bash
pip install umap-learn
```

### Clasificador LLM por prompt opcional

`src/suicidality/llm_prompt_classifier.py` ofrece una interfaz experimental para
clasificar con un prompt y un generador inyectado. Tambien incluye
`HuggingFacePromptLLMClassifier`, que carga un modelo generativo local de forma
diferida. No requiere una API pagada y no participa en la comparacion principal.

### Pruebas

```bash
pytest -q
```
