# Parte B - Implementacion

## Algoritmo de clasificacion
Se usa un modelo base de regresion logistica sobre TF-IDF y banderas lexicas. El modelo produce probabilidad y se evalua con ROC-AUC. Se incluye un fast-path para frases criticas.

## Pruebas unitarias
Las pruebas cubren carga de datos, lexer, preprocesamiento y pipeline basico.

## Optimizacion
- TF-IDF con ngramas (1,2) y min_df configurable.
- Balance de clases con class_weight.
- Semilla fija para reproducibilidad.

## Resultados
Los resultados se guardan en reports/metrics.json y la curva ROC en reports/roc.png.
