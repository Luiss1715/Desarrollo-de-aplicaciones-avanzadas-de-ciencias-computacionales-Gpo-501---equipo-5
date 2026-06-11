# Parte B - Implementacion

## Algoritmo de clasificacion
Se usa un modelo base de regresion logistica sobre TF-IDF y banderas lexicas. El modelo siempre produce la probabilidad y se evalua con ROC-AUC; las frases criticas no fuerzan automaticamente una clasificacion.

## Pruebas unitarias
Las pruebas cubren carga de datos, lexer, preprocesamiento y pipeline basico.

## Optimizacion
- TF-IDF con ngramas (1,2) y min_df configurable.
- Balance de clases con class_weight.
- Semilla fija para reproducibilidad.

## Resultados
ROC-AUC obtenido (split 80/20, seed 42): 0.7769.

Artefactos generados:
- reports/metrics.json
- reports/roc.png
