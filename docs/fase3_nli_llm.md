# Fase 3: NLI, transformers y espacio latente

## Métodos evaluados

La comparación usa tres métodos sobre el mismo conjunto de prueba:

| Método | Representación | Clasificador |
| --- | --- | --- |
| `phase2` | TF-IDF y banderas léxicas | Regresión logística |
| `phase3_latent` | Embeddings de Transformer | MLP entrenada |
| `phase3_nli` | Texto completo | NLI zero-shot |

La entrada de cada método es la concatenación de `title` y `text`. Cada salida
contiene una etiqueta binaria y un score continuo entre 0 y 1.

## NLI zero-shot

NLI evalúa la relación entre una premisa y una hipótesis. En esta implementación,
la publicación es la premisa y el modelo MNLI compara las etiquetas `suicidal
ideation` y `no suicidal ideation` usando la plantilla `This text expresses {}`.
El score de `suicidal ideation` se compara con un umbral de 0.5 por defecto.

El modelo predeterminado es `facebook/bart-large-mnli`. NLI no necesita
entrenamiento con el dataset del proyecto, pero su resultado depende de cómo se
formulan las etiquetas y de la capacidad del modelo para interpretar textos
largos, ambiguos, irónicos o citados.

## Transformers, espacio latente y MLP

`sentence-transformers/all-MiniLM-L6-v2` convierte cada publicación en un vector
denso. Esos embeddings forman el espacio latente: textos semánticamente parecidos
pueden quedar próximos aunque no compartan las mismas palabras.

La MLP recibe los embeddings y aprende una frontera de decisión con dos capas
ocultas, activaciones ReLU y dropout. Se entrena con `BCEWithLogitsLoss`. El
comparador crea una validación interna agrupada por usuario para calibrar la MLP
y después la reentrena con todo el conjunto de entrenamiento. Las métricas
finales se calculan únicamente sobre el CSV de prueba separado.

La proyección PCA en `reports/latent_space_pca.png` permite inspeccionar la
separación de clases en dos dimensiones. PCA sirve para visualización; no es la
representación usada para entrenar la MLP.

## Clasificador LLM por prompt

`src/suicidality/llm_prompt_classifier.py` incluye un experimento opcional por
prompt. La clase `PromptLLMClassifier` acepta una función generadora inyectada,
por lo que se puede probar sin API ni credenciales. La clase
`HuggingFacePromptLLMClassifier` carga de forma diferida un modelo generativo
local de Hugging Face.

Este componente exige que la respuesta contenga JSON con `label` y `score`.
No forma parte de la tabla principal porque los métodos obligatorios de Fase 3
son NLI y embeddings con MLP. Un modelo generativo también puede incumplir el
formato solicitado, por lo que requiere validación adicional antes de usarse
como clasificador.

## Protocolo de comparación

El comando principal es:

```bash
python -m suicidality.compare_phase2_phase3 --csv data_train.csv --test-csv data_test_fold2.csv --reports-dir reports --models-dir models
```

El comparador crea una validación interna agrupada por `user_id` dentro de
`data_train.csv`. Esa validación calibra umbrales, selecciona la mejor época de
la MLP y entrena el stacking. Luego los modelos base se reentrenan con todo
`data_train.csv`; `data_test_fold2.csv` se usa solo para el reporte final. Para
cada texto de prueba guarda la etiqueta real, la predicción binaria y el score.
Después calcula:

- `TP`: textos suicidas clasificados como suicidas.
- `TN`: textos no suicidas clasificados como no suicidas.
- `FP`: textos no suicidas clasificados como suicidas.
- `FN`: textos suicidas clasificados como no suicidas.
- `TPR = TP / (TP + FN)`.
- `FPR = FP / (FP + TN)`.
- `AUC = (1 + TPR - FPR) / 2`, fórmula solicitada por el protocolo.
- `ROC-AUC`, calculada con scores continuos cuando el conjunto contiene ambas clases.

La AUC del protocolo resume un único punto de operación definido por el umbral.
ROC-AUC evalúa el ordenamiento de scores a través de todos los umbrales. No son
intercambiables y ambas se conservan en los reportes.

## Interpretación de resultados

La comparación real se guarda en `reports/comparison_table.csv` con esta forma:

| method | TP | TN | FP | FN | TPR | FPR | AUC | ROC-AUC | comentario |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `phase2` | 14 | 21 | 4 | 12 | 0.5385 | 0.1600 | 0.6892 | 0.7769 | TF-IDF, banderas léxicas y regresión logística |
| `phase3_latent` | 17 | 17 | 8 | 9 | 0.6538 | 0.3200 | 0.6669 | 0.6831 | Embeddings de Transformer y MLP |
| `phase3_nli` | 25 | 3 | 22 | 1 | 0.9615 | 0.8800 | 0.5408 | 0.6708 | NLI zero-shot |

Estos resultados corresponden a `DataSet.csv`, split estratificado 80/20, semilla
42 y umbral 0.5. Fase 2 obtuvo la mayor AUC del protocolo y ROC-AUC, además del
menor número de falsos positivos. La MLP redujo los falsos negativos de 12 a 9,
pero duplicó los falsos positivos de 4 a 8. NLI detectó 25 de 26 positivos y dejó
solo un falso negativo, a costa de clasificar casi todos los textos como
positivos: produjo 22 falsos positivos y un FPR de 0.88.

Fase 2 es más barata e interpretable: permite inspeccionar términos, banderas y
coeficientes. La MLP puede aprender patrones semánticos menos evidentes, pero
requiere generar embeddings y su decisión es menos interpretable. NLI evita
entrenamiento local, aunque tiene mayor costo computacional y sensibilidad a la
formulación de etiquetas. Si Fase 3 no supera a Fase 2 en AUC o falsos negativos,
sigue siendo útil como referencia semántica y para estudiar errores que TF-IDF
no representa bien.

## Ejecución con GPU

En una computadora con PyTorch configurado para CUDA, la comparación puede
ejecutarse con:

```bash
python -m suicidality.compare_phase2_phase3 --csv data_train.csv --test-csv data_test_fold2.csv --reports-dir reports --models-dir models --device cuda --nli-device 0
```

La GPU reduce principalmente el tiempo de embeddings, entrenamiento de la MLP y
NLI. Las métricas solo son comparables si se conservan el dataset, semilla,
split, umbral y modelos.

## Artefactos

Una ejecución completa genera:

- `reports/metrics.json` y `reports/roc.png` para Fase 2.
- `reports/phase2_predictions.csv`.
- `reports/phase3_latent_predictions.csv`.
- `reports/phase3_nli_predictions.csv`.
- `reports/comparison_metrics.json`.
- `reports/comparison_table.csv`.
- `reports/latent_space_pca.png`.
- `models/phase3_thresholds.json`.
- `models/phase3_ensemble.joblib`.
- `models/pipeline.joblib`, `models/latent_nn.pt` y `models/latent_config.json`.

UMAP es opcional. Se puede instalar con `pip install umap-learn` y generar con:

```bash
python -m suicidality.latent_visualization --csv DataSet.csv
```
