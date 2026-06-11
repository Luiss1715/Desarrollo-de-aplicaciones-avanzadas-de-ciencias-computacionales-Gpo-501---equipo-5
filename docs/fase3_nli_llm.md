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
comparador crea un split estratificado común, entrena la MLP únicamente con la
partición de entrenamiento y calcula sus métricas sobre la misma partición de
prueba usada por Fase 2 y NLI.

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
python -m suicidality.compare_phase2_phase3 --csv DataSet.csv --reports-dir reports --models-dir models
```

El comparador usa un split estratificado 80/20 con semilla 42 por defecto. Para
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
| `phase2` | generado al ejecutar | generado al ejecutar | generado al ejecutar | generado al ejecutar | generado al ejecutar | generado al ejecutar | generado al ejecutar | generado al ejecutar | TF-IDF, banderas léxicas y regresión logística |
| `phase3_latent` | generado al ejecutar | generado al ejecutar | generado al ejecutar | generado al ejecutar | generado al ejecutar | generado al ejecutar | generado al ejecutar | generado al ejecutar | Embeddings de Transformer y MLP |
| `phase3_nli` | generado al ejecutar | generado al ejecutar | generado al ejecutar | generado al ejecutar | generado al ejecutar | generado al ejecutar | generado al ejecutar | generado al ejecutar | NLI zero-shot |

No se debe concluir que Fase 3 mejora solo porque usa modelos más grandes. La
conclusión debe revisar especialmente los falsos negativos, porque representan
casos de riesgo no detectados. Los falsos positivos también importan porque
pueden generar alertas innecesarias.

Fase 2 es más barata e interpretable: permite inspeccionar términos, banderas y
coeficientes. La MLP puede aprender patrones semánticos menos evidentes, pero
requiere generar embeddings y su decisión es menos interpretable. NLI evita
entrenamiento local, aunque tiene mayor costo computacional y sensibilidad a la
formulación de etiquetas. Si Fase 3 no supera a Fase 2 en AUC o falsos negativos,
sigue siendo útil como referencia semántica y para estudiar errores que TF-IDF
no representa bien.

## Artefactos

Una ejecución completa genera:

- `reports/metrics.json` y `reports/roc.png` para Fase 2.
- `reports/phase2_predictions.csv`.
- `reports/phase3_latent_predictions.csv`.
- `reports/phase3_nli_predictions.csv`.
- `reports/comparison_metrics.json`.
- `reports/comparison_table.csv`.
- `reports/latent_space_pca.png`.
- `models/pipeline.joblib`, `models/latent_nn.pt` y `models/latent_config.json`.

UMAP es opcional. Se puede instalar con `pip install umap-learn` y generar con:

```bash
python -m suicidality.latent_visualization --csv DataSet.csv
```
