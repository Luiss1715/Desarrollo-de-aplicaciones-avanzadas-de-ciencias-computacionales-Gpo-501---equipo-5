# Fase 3: comparacion de siete clasificadores

La Fase 3 compara siete metodos sobre el mismo conjunto de prueba. Los primeros
cuatro usan exactamente los mismos embeddings de Transformer, por lo que sus
diferencias se deben al clasificador y no a la representacion.

| Metodo | Representacion | Clasificador |
| --- | --- | --- |
| `phase3_latent_mlp` | Embeddings de Transformer | MLP en PyTorch |
| `phase3_latent_logreg` | Embeddings de Transformer | Regresion logistica |
| `phase3_latent_svm` | Embeddings de Transformer | SVM lineal |
| `phase3_latent_random_forest` | Embeddings de Transformer | Random Forest |
| `phase3_nli_zero_shot` | Texto y dos etiquetas NLI | NLI zero-shot |
| `phase3_nli_supervised` | Puntajes de seis hipotesis NLI | Regresion logistica supervisada |
| `phase3_ensemble` | Scores de los seis modelos anteriores | Stacking con regresion logistica |

`phase2` tambien aparece en la tabla final, pero se conserva como linea base y
no se cuenta entre los siete clasificadores de Fase 3.

## NLI supervisado

El segundo metodo NLI no usa directamente el umbral del modelo zero-shot.
Primero obtiene puntajes para seis hipotesis semanticas, por ejemplo
`suicidal ideation`, `desire to die`, `hopelessness` y
`an ordinary daily experience`. Despues entrena una regresion logistica con las
etiquetas reales para aprender como combinar esas senales.

La validacion interna se usa para calibrar el umbral de cada clasificador. Los
modelos finales se reentrenan con todo `data_train.csv`, y
`data_test_fold2.csv` se usa exclusivamente para el reporte final.

## Ejecucion

```bash
python -m suicidality.compare_phase2_phase3 \
  --csv data_train.csv \
  --test-csv data_test_fold2.csv \
  --reports-dir reports \
  --models-dir models
```

La ejecucion genera una fila y un CSV de predicciones para cada metodo. Tambien
guarda los modelos clasicos del espacio latente:

- `models/latent_logistic_regression.joblib`
- `models/latent_linear_svm.joblib`
- `models/latent_random_forest.joblib`
- `models/nli_supervised.joblib`
- `models/phase3_ensemble.joblib`
- `models/phase3_thresholds.json`

El NLI supervisado procesa el conjunto de entrenamiento completo con el modelo
MNLI, por lo que esta parte puede tardar considerablemente mas que zero-shot
sobre solamente el conjunto de prueba.

## Graficas comparativas

Al finalizar, el comparador genera seis graficas para interpretar resultados:

- `model_metrics_comparison.png`: TPR, FPR, AUC y ROC-AUC por modelo.
- `confusion_matrices.png`: TP, TN, FP y FN de todos los clasificadores.
- `roc_curves_comparison.png`: curvas ROC conjuntas.
- `precision_recall_curves.png`: curvas Precision-Recall conjuntas.
- `fp_fn_tradeoff.png`: balance entre falsas alertas y casos no detectados.
- `threshold_sensitivity.png`: efecto del umbral sobre la AUC del protocolo.

Se pueden regenerar usando los resultados existentes, sin ejecutar nuevamente
los modelos:

```bash
python -m suicidality.fase3.model_visualizations \
  --reports-dir reports \
  --thresholds models/phase3_thresholds.json
```
